import os
import tensorflow as tf
from data_interface.dataset_wrapper import DatasetInterfaceWrapper
from idas import utils
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.routine_callback import RoutineCallback
from idas.callbacks.early_stopping_callback import EarlyStoppingCallback, EarlyStoppingException, NeedForTestException
import config_file
from architectures.AutoEncoder import AE
from idas.losses.tf_losses import weighted_softmax_cross_entropy
from idas.utils import ProgressBar
import random
import time
import errno


class AEModel(DatasetInterfaceWrapper):
    def __init__(self, run_id=None):
        """
        Auto-encoder model. It defines the network architecture and the functions for train, test, etc.
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py, discriminate different training process
        """

        FLAGS = config_file.define_flags()

        self.ae_run_id = FLAGS.AE_RUN_ID if (run_id is None) else run_id
        self.num_threads = FLAGS.num_threads
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.CUDA_VISIBLE_DEVICE)

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        self.batch_size = FLAGS.b_size
        self.nz_latent = FLAGS.nz_latent  # to control the dimension of latent code
        self.n_ch_out = FLAGS.n_ch_out

        # -----------------------------
        # Data
        self.label_input_size = FLAGS.input_size
        self.acdc_data_path = FLAGS.acdc_data_path  # list of path for the training and validation files:
        self.cub_data_path = FLAGS.cub_data_path
        self.augment = FLAGS.augment  # perform data augmentation
        self.standardize = FLAGS.standardize  # perform data standardization

        # -----------------------------
        # Report
        # path to save checkpoints and graph
        self.last_checkpoint_dir = './ae_results/checkpoints/' + FLAGS.AE_RUN_ID
        self.checkpoint_dir = './ae_results/checkpoints/' + FLAGS.AE_RUN_ID
        self.graph_dir = './ae_results/graphs/' + FLAGS.AE_RUN_ID + '/convnet'
        self.history_log_dir = './ae_results/history_logs/' + FLAGS.AE_RUN_ID

        # verbosity
        self.skip_step = FLAGS.skip_step  # frequency of batch report
        self.train_summaries_skip = FLAGS.train_summaries_skip  # number of skips before writing train summaries
        self.tensorboard_verbose = FLAGS.tensorboard_verbose  # (bool) save also layers weights at the end of epoch

        # -----------------------------
        # Callbacks
        # init the list of callbacks to be called and relative arguments
        self.callbacks = []
        self.callbacks_kwargs = {'history_log_dir': self.history_log_dir}
        self.callbacks.append(RoutineCallback())  # routine callback always runs
        # Early stopping callback:
        self.callbacks_kwargs['es_loss'] = None
        self.best_val_loss = tf.Variable(1e10, dtype=tf.float32, trainable=False, name='best_val_loss')
        self.update_best_val_loss = self.best_val_loss.assign(
            tf.placeholder(tf.float32, None, name='best_val_loss_value'), name='update_best_val_loss')
        self.callbacks_kwargs['test_on_minimum'] = True
        self.callbacks.append(EarlyStoppingCallback(min_delta=1e-5, patience=2000))

        # -----------------------------
        # Other settings
        # Define global step for training e validation and counter for global epoch:
        self.g_train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_train_step')
        self.g_valid_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_validation_step')
        self.g_test_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_test_step')
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # define their update operations
        up_value = tf.placeholder(tf.int32, None, name='update_value')
        self.update_g_train_step = self.g_train_step.assign(up_value, name='update_g_train_step')
        self.update_g_valid_step = self.g_valid_step.assign(up_value, name='update_g_valid_step')
        self.update_g_test_step = self.g_test_step.assign(up_value, name='update_g_test_step')
        self.increase_g_epoch = self.g_epoch.assign_add(1, name='increase_g_epoch')

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.label_input_size,
                         num_threads=self.num_threads)

    def build(self):
        """
        Build the computation graph for the auto-encoder
        :return:
        """
        print('Building the computation graph...\nAE_RUN_ID = \033[94m{0}\033[0m'.format(self.ae_run_id))
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def get_data(self):
        """
        Define the dataset iterators for the reconstruction task.
        They will be used in define_model().
        :return:
        """
        self.global_seed = tf.placeholder(tf.int64, shape=())
        # self.train_init, self.valid_init, self.test_init, self.texture_data, self.label_data = \
        #     super(AEModel, self).get_acdc_texture_shape_data(data_path=self.acdc_data_path,
        #                                                      repeat=False, seed=self.global_seed)

        self.train_init, self.valid_init, self.test_init, \
        self.image_data, self.label_data, self.texture_data = \
            super(AEModel, self).get_cub_disc_data(data_path=self.cub_data_path,
                                                   repeat=False, seed=self.global_seed)

    def define_model(self):
        """
        Define the network architecture.
        Notice that we want to train 2 auto-encoder separately for texture reconstruction and label reconstruction.
        So we must define two AE with reuse=False.
        :return:
        """

        texture_ae = AE(n_out=self.n_ch_out, is_training=self.is_training, n_filters=64, name='texture_ae', trainable=True)
        self.texture_output = texture_ae.build(self.texture_data)

        label_ae = AE(n_out=self.n_ch_out, is_training=self.is_training, n_filters=64, name='label_ae', trainable=True)
        self.label_output = label_ae.build(self.label_data, reuse=False)

    def define_losses(self):
        """
        Define loss function for each AE
        :return:
        """
        with tf.variable_scope('texture_rec_loss'):
            self.texture_rec_loss = tf.reduce_mean(tf.abs(self.texture_data - self.texture_output))

        with tf.variable_scope('label_rec_loss'):
            self.label_rec_loss = tf.reduce_mean(tf.abs(self.label_data - self.label_output))

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        :return:
        """

        def _train_op_wrapper(loss_function, optimizer, clip_grads=False, clip_value=5.0, var_list=None):
            """ define optimizer and train op with gradient clipping. """

            # define update_ops to update batch normalization population statistics
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                # gradient clipping for stability:
                gradients, variables = zip(*optimizer.compute_gradients(loss_function, var_list))
                if clip_grads:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
                # train op:
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.g_train_step)

            return train_op

        clip = False
        optimizer = tf.train.AdamOptimizer(self.lr)

        texture_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "texture_ae")
        self.train_op_texture = _train_op_wrapper(self.texture_rec_loss, optimizer, clip, var_list=texture_vars)

        label_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "label_ae")
        self.train_op_label = _train_op_wrapper(self.label_rec_loss, optimizer, clip, var_list=label_vars)

        self.global_train_op = tf.group(self.train_op_texture, self.train_op_label)

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        :return:
        """
        with tf.variable_scope('texutre_mse'):
            self.texture_mse = tf.reduce_mean(tf.squared_difference(self.texture_output, self.texture_data))

        with tf.variable_scope('label_mse'):
            self.label_mse = tf.reduce_mean(tf.squared_difference(self.label_output, self.label_data))

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        :return:
        """

        with tf.name_scope('texture_reconstruction'):
            train_texture_rec = tf.summary.scalar('train/texture_rec_loss', self.texture_rec_loss)
            valid_texture_rec = tf.summary.scalar('validation/texture_rec_loss', self.texture_rec_loss)

        with tf.name_scope('label_reconstruction'):
            train_label_rec = tf.summary.scalar('train/label_rec_loss', self.label_rec_loss)
            valid_label_rec = tf.summary.scalar('validation/label_rec_loss', self.label_rec_loss)

        self.train_summary_op = tf.summary.merge([train_texture_rec, train_label_rec])
        self.val_summary_op = tf.summary.merge([valid_texture_rec, valid_label_rec])

        # ---- #
        if self.tensorboard_verbose:
            _vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in _vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def _train_all_op(self, sess, writer, step):
        _, t_loss, l_loss, train_summary = sess.run([self.global_train_op,
                                                            self.texture_rec_loss,
                                                            self.label_rec_loss,
                                                            self.train_summary_op],
                                                           feed_dict={self.is_training: True})

        if random.randint(0, self.train_summaries_skip) == 0:
            writer.add_summary(train_summary, global_step=step)

        return t_loss, l_loss

    def train_one_epoch(self, sess, iterator_init_list, writer, step, caller, seed):
        """
        train the model for one epoch.
        :param sess:
        :param iterator_init_list:
        :param writer:
        :param step:
        :param caller:
        :param seed:
        :return:
        """
        start_time = time.time()

        # setup progress bar
        try:
            self.progress_bar.attach()
        except:
            self.progress_bar = ProgressBar(update_delay=20)
            self.progress_bar.attach()

        self.progress_bar.monitor_progress()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init, feed_dict={self.global_seed: seed})

        total_texture_loss = 0
        total_label_loss = 0
        n_batches = 0

        try:
            while True:
                self.progress_bar.monitor_progress()

                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                texture_loss, label_loss = self._train_all_op(sess, writer, step)
                total_texture_loss += texture_loss
                total_label_loss += label_loss
                step += 1

                n_batches += 1
                if (n_batches % self.skip_step) == 0:
                    print('\r  ...training over batch {1}: {0} batch_texture_loss = {2:.4f}\tbatch_label_loss = {3:.4f} {0}'
                          .format(' ' * 3, n_batches, texture_loss, label_loss), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)

                # print("Batch: ", n_batches)
                # print("total_texture_loss: ", total_texture_loss)
                # print("total_label_loss: ", total_label_loss)

        except tf.errors.OutOfRangeError:
            texture_avg_loss = total_texture_loss / n_batches
            label_avg_loss = total_label_loss / n_batches
            delta_t = time.time() - start_time

        # update global epoch counter:
        sess.run(self.increase_g_epoch)
        sess.run(self.update_g_train_step, feed_dict={'update_value:0': step})

        # detach progress bar and update last time of arrival:
        self.progress_bar.detach()
        self.progress_bar.update_lta(delta_t)

        print('\033[31m  TRAIN\033[0m:{0}{0} average texture loss = {1:.4f} {0}, average label loss = {2:.4f} {0}. '
              'Took: {3:.3f} seconds.'
              .format(' ' * 3, texture_avg_loss, label_avg_loss, delta_t))
        return step

    def _eval_all_op(self, sess, writer, step):
        t_loss, l_loss, valid_summary = sess.run([self.texture_rec_loss,
                                                  self.label_rec_loss,
                                                  self.val_summary_op],
                                                 feed_dict={self.is_training: False})

        writer.add_summary(valid_summary, global_step=step)
        return t_loss, l_loss

    def eval_once(self, sess, iterator_init_list, writer, step, caller):
        """
         Eval the model once
        :param sess:
        :param iterator_init_list:
        :param writer:
        :param step:
        :param caller:
        :return:
        """
        start_time = time.time()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init)

        total_texture_loss = 0
        total_label_loss = 0
        n_batches = 0

        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                texture_loss, label_loss = self._eval_all_op(sess, writer, step)
                total_texture_loss += texture_loss
                total_label_loss += label_loss
                step += 1

                n_batches += 1
                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:

            texture_avg_loss = total_texture_loss / n_batches
            label_avg_loss = total_label_loss / n_batches

            delta_t = time.time() - start_time

            pass

        # update global epoch counter:
        sess.run(self.update_g_valid_step, feed_dict={'update_value:0': step})

        print('\033[31m  VALIDATION\033[0m:  average texture loss = {1:.4f} {0}, average label loss = {2:.4f} {0}.'
              ' Took: {3:.3f} seconds'
              .format(' ' * 3, texture_avg_loss, label_avg_loss, delta_t))
        return step, texture_avg_loss, label_avg_loss

    def test_once(self, sess, test_init, writer, step, caller):
        """
         Test the model once
        :param sess:
        :param test_init:
        :param writer:
        :param step:
        :param caller:
        :return:
        """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(test_init)

        total_texture_mse = 0
        total_label_mse = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                texture_mse, label_mse = sess.run([self.texture_mse, self.label_mse],
                                                  feed_dict={self.is_training: False})
                total_texture_mse += texture_mse
                total_label_mse += label_mse

                # writer.add_summary(images_summaries, global_step=step)

                n_batches += 1

        except tf.errors.OutOfRangeError:

            avg_texture_mse = total_texture_mse / n_batches
            avg_label_mse = total_label_mse / n_batches
            delta_t = time.time() - start_time

            step += 1

            # value = summary_pb2.Summary.Value(tag="y_TEST/test/dice_3channels_avg", simple_value=avg_dice)
            # summary = summary_pb2.Summary(value=[value])
            # writer.add_summary(summary, global_step=step)
            pass

        # update global epoch counter:
        sess.run(self.update_g_test_step, feed_dict={'update_value:0': step})

        print('\033[31m  TEST\033[0m:{0}{0} \033[1;33m average texture mse = {1:.4f}, average label mse = {2:.4f}'
              '\033[0m on \033[1;33m{3}\033[0m batches '
              '{0} Took: {4:.3f} seconds'.format(' ' * 3, avg_texture_mse, avg_label_mse, n_batches, delta_t))
        return step

    def train(self, n_epochs):
        """
        The train function alternates between training one epoch and evaluating
        :param n_epochs:
        :return:
        """
        print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
        print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
        print("Tensorboard dir: \033[94m{0}\033[0m".format(self.graph_dir))
        print("Data augmentation: \033[94m{0}\033[0m, Data standardization: \033[94m{1}\033[0m."
              .format(self.augment, self.standardize))

        utils.safe_mkdir(self.checkpoint_dir)
        utils.safe_mkdir(self.history_log_dir)
        writer = tf.summary.FileWriter(self.graph_dir, tf.get_default_graph())

        # config for the session: allow growth for GPU to avoid OOM when other processes are running
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()  # keep_checkpoint_every_n_hours=2
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.last_checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            trained_epochs = self.g_epoch.eval()  # global step is also saved in checkpoint
            print("Model already trained for \033[94m{0}\033[0m epochs.".format(trained_epochs))
            t_step = self.g_train_step.eval()  # global step for train
            v_step = self.g_valid_step.eval()  # global step for validation
            test_step = self.g_test_step.eval()  # global step for test

            # Define a caller to call the callbacks
            self.callbacks_kwargs.update({'sess': sess, 'cnn': self})
            caller = tf_callbacks.ChainCallback(callbacks=self.callbacks)
            caller.on_train_begin(training_state=True, **self.callbacks_kwargs)

            # trick to find performance bugs: this will raise an exception if any new node is inadvertently added to the
            # graph. This will ensure that I don't add many times the same node to the graph (which could be expensive):
            tf.get_default_graph().finalize()

            # saving callback:
            self.callbacks_kwargs['es_loss'] = 100  # some random initialization

            for epoch in range(n_epochs):
                ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(
                    epoch + 1)
                print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - {1} : '.format(ep_str, self.ae_run_id))

                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                global_ep = sess.run(self.g_epoch)
                self.callbacks_kwargs['es_loss'] = sess.run(self.best_val_loss)

                seed = global_ep

                # TRAINING ------------------------------------------
                iterator_init_list = [self.train_init]
                t_step = self.train_one_epoch(sess, iterator_init_list, writer, t_step, caller, seed)

                # VALIDATION ------------------------------------------
                if global_ep >= 15 or not ((global_ep + 1) % 5):  # when to evaluate the model
                    iterator_init_list = [self.valid_init]
                    v_step, val_t_loss, val_l_loss = self.eval_once(sess, iterator_init_list, writer, v_step, caller)
                    self.callbacks_kwargs['es_loss'] = val_l_loss + val_t_loss
                    sess.run(self.update_best_val_loss, feed_dict={'best_val_loss_value:0': val_l_loss + val_t_loss})

                # writing summary for the weights:
                if self.tensorboard_verbose and (global_ep % 20 == 0):
                    summary = sess.run(self.weights_summary)
                    writer.add_summary(summary, global_step=t_step)

                try:
                    caller.on_epoch_end(training_state=True, **self.callbacks_kwargs)
                except EarlyStoppingException:
                    utils.print_yellow_text('\nEarly stopping...\n')
                    break
                except NeedForTestException:
                    # found new best model, save it
                    saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

            caller.on_train_end(training_state=True, **self.callbacks_kwargs)

            # end of the training: save the current weights in a new sub-directory
            utils.safe_mkdir(self.checkpoint_dir + '/last_model')
            saver.save(sess, self.checkpoint_dir + '/last_model/checkpoint', t_step)

            # load best model and do a test:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            _ = self.test_once(sess, self.test_init, writer, test_step, caller)

        writer.close()

    def test(self, input_texture, input_label):
        """
        Test the model on input_data
        :param input_texture:
        :param input_label
        :return:
        """
        if self.standardize:
            print('Remember to standardize your data!')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:

                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Returning: (texture_rec, label_rec)')
                output = sess.run([self.texture_output, self.label_output],
                                  feed_dict={self.texture_data: input_texture,
                                             self.label_data: input_label,
                                             self.is_training: False})
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = AEModel()
    model.build()
    model.train(n_epochs=20)
