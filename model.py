import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from idas import utils
import tensorflow as tf
from data_interface.dataset_wrapper import DatasetInterfaceWrapper
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.routine_callback import RoutineCallback
from idas.callbacks.early_stopping_callback import EarlyStoppingCallback, EarlyStoppingException, NeedForTestException
import config_file
from architectures.mask_discriminator import MaskDiscriminator
from architectures.sdnet import SDNet
from architectures.AutoEncoder import AE
from idas.metrics.tf_metrics import dice_coe
from idas.losses.tf_losses import weighted_softmax_cross_entropy, generalized_dice_loss
from tensorflow.core.framework import summary_pb2
import errno
from idas.utils import ProgressBar
import random
import numpy as np


class Model(DatasetInterfaceWrapper):
    def __init__(self, run_id=None):
        """
        General model. It defines the network architecture and the functions for train, test, etc.

        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        """

        FLAGS = config_file.define_flags()

        self.run_id = FLAGS.RUN_ID if (run_id is None) else run_id
        self.num_threads = FLAGS.num_threads

        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.CUDA_VISIBLE_DEVICE)

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        self.batch_size = FLAGS.b_size
        self.nz_latent = FLAGS.nz_latent  # for modality encoder
        self.n_anatomical_masks = FLAGS.n_anatomical_masks
        self.label_ae_filters = FLAGS.label_ae_filters
        self.texture_ae_filters = FLAGS.texture_ae_filters
        self.disc_times = FLAGS.disc_times

        # -----------------------------
        # Data

        # data specifics
        self.input_size = FLAGS.input_size  # not including texture
        self.n_classes = FLAGS.n_classes  # originally used for segmentation

        # ACDC data set
        self.acdc_data_path = FLAGS.acdc_data_path  # list of path for the training and validation files:

        # data pre-processing
        self.augment = FLAGS.augment  # perform data augmentation
        self.standardize = FLAGS.standardize  # perform data standardization

        # -----------------------------
        # Report

        # path to save checkpoints and graph
        self.last_checkpoint_dir = './results/checkpoints/' + FLAGS.RUN_ID
        self.ae_checkpoint_dir = './ae_results/checkpoints/' + FLAGS.AE_RUN_ID
        self.checkpoint_dir = './results/checkpoints/' + FLAGS.RUN_ID
        self.graph_dir = './results/graphs/' + FLAGS.RUN_ID + '/convnet'
        self.history_log_dir = './results/history_logs/' + FLAGS.RUN_ID
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
                         input_size=self.input_size,
                         num_threads=self.num_threads)

    def build(self):
        """ Build the computation graph """
        print('Building the computation graph...\nRUN_ID = \033[94m{0}\033[0m'.format(self.run_id))
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def get_data(self):
        """ Define the dataset iterators for each task (supervised, unsupervised and mask discriminator)
        They will be used in define_model().
        """

        self.global_seed = tf.placeholder(tf.int64, shape=())

        self.disc_train_init, self.disc_valid_init, self.disc_test_init, \
        self.disc_image_data, self.disc_label_data, self.disc_label_data_oh, self.disc_texture_data = \
            super(Model, self).get_acdc_disc_data(data_path=self.acdc_data_path, repeat=False, seed=self.global_seed)

        self.unsup_train_init, self.unsup_valid_init, \
        self.unsup_input_data, self.unsup_output_data = \
            super(Model, self).get_acdc_unsup_data(data_path=self.acdc_data_path, repeat=True, seed=self.global_seed)

    def define_model(self):
        """ Define the network architecture.
        Notice that, since we want to share the weights across different tasks we must define one SDNet for every task
        with reuse=True. Then, we have:

          - sdnet_disc: model for the adversarial discriminator. 2 discriminator are used separately for texture
                        reconstruction from texture/image and label reconstruction from label/image.
          - sdnet_unsup: model for the unsupervised task of decomposing the image in anatomical and modality dependent
                        factors (s and z, respectively). The model is trained to reconstruct the input image using them.
        """

        sdnet_disc = SDNet(self.n_anatomical_masks, self.nz_latent, self.n_classes, self.is_training,
                           name='Model')  # all prediction masks are false
        sdnet_disc = sdnet_disc.build(self.disc_image_data)

        sdnet_unsup = SDNet(self.n_anatomical_masks, self.nz_latent, self.n_classes, self.is_training,
                            name='Model')
        sdnet_unsup = sdnet_unsup.build(self.unsup_input_data, reuse=True)

        # the auto-encoder-sure model
        texture_ae = AE(n_out=1, is_training=False, n_filters=self.texture_ae_filters, name='texture_ae', trainable=False)

        label_ae = AE(n_out=1, is_training=False, n_filters=self.label_ae_filters, name='label_ae', trainable=False)

        # - - - - - - -
        # define tensors for the losses:

        # pre-trained
        self.texture_output = texture_ae.build(self.disc_texture_data)
        self.label_output = label_ae.build(self.disc_label_data)

        # unsup pathway
        self.unsup_reconstruction = sdnet_unsup.get_input_reconstruction()  # used in loss
        self.unsup_z_mean, self.unsup_z_logvar, self.unsup_sampled_z = sdnet_unsup.get_z_distribution()
        self.unsup_z_regress = sdnet_unsup.get_z_sample_estimate()

        # disc pathway
        self.anatomy_code = sdnet_disc.get_anatomy_code()

        print("label_code.shape: ", self.anatomy_code.shape)  # compare with the training size
        self.label_rec_decoder = label_ae.build_decoder(self.anatomy_code, reuse=True)
        self.label_rec = label_ae.build_output(self.label_rec_decoder, reuse=True)

        texture_code_shape = tf.constant(np.array([self.batch_size,
                                                   np.power(self.nz_latent, 1/3),
                                                   np.power(self.nz_latent, 1/3),
                                                   np.power(self.nz_latent, 1/3)], dtype=np.int32))
        self.texture_code = tf.reshape(self.unsup_sampled_z, texture_code_shape)
        self.texture_code = tf.tile(self.texture_code, [1, 1, 1, self.texture_ae_filters*8])
        print("texture_code.shape: ", self.texture_code.shape)
        self.texture_rec_decoder = texture_ae.build_decoder(self.texture_code, reuse=True)

        self.texture_rec = label_ae.build_output(self.texture_rec_decoder, reuse=True)

        self.disc_hard_anatomy = sdnet_disc.get_hard_anatomy()
        self.disc_soft_anatomy = sdnet_disc.get_soft_anatomy()

        # - - - - - - -
        # build Label Discriminator (Least Square GAN)
        with tf.variable_scope('LabelDiscriminator'):
            l_model_real = MaskDiscriminator(self.is_training, n_filters=64, out_mode='scalar')
            l_model_real = l_model_real.build(self.label_output, reuse=False)

            l_model_fake = MaskDiscriminator(self.is_training, n_filters=64, out_mode='scalar')
            l_model_fake = l_model_fake.build(self.label_rec, reuse=True)

            self.label_disc_real = l_model_real.get_prediction()
            self.label_disc_fake = l_model_fake.get_prediction()

        with tf.variable_scope('TextureDiscriminator'):
            t_model_real = MaskDiscriminator(self.is_training, n_filters=64, out_mode='scalar')
            t_model_real = t_model_real.build(self.texture_output, reuse=False)

            t_model_fake = MaskDiscriminator(self.is_training, n_filters=64, out_mode='scalar')
            t_model_fake = t_model_fake.build(self.texture_rec, reuse=True)

            self.texture_disc_real = t_model_real.get_prediction()
            self.texture_disc_fake = t_model_fake.get_prediction()

    def define_losses(self):
        """
        Define loss function for each task.
        """
        # _______
        # Reconstruction loss:
        with tf.variable_scope('Reconstruction_loss'):
            self.unsup_image_rec_loss = tf.reduce_mean(tf.abs(self.unsup_reconstruction - self.unsup_output_data))
            self.unsup_z_regress_loss = tf.reduce_mean(tf.abs(self.unsup_z_regress - self.unsup_sampled_z))
            self.texture_rec_loss = tf.reduce_mean(tf.abs(self.texture_rec - self.texture_output))
            self.label_rec_loss = tf.reduce_mean(tf.abs(self.label_rec - self.label_output))

        # _______
        # KL Divergence loss:
        with tf.variable_scope('KL_divergence_loss'):
            kl_i = 1.0 + self.unsup_z_logvar - tf.square(self.unsup_z_mean) - tf.exp(self.unsup_z_logvar)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_i, 1)
            self.kl_div_loss = tf.reduce_mean(kl_div_loss)

        # _______
        # Mask Discriminator loss:
        # this is a LeastSquare GAN: use MSE as loss
        with tf.variable_scope('LabelDiscriminator_loss'):
            self.label_adv_disc_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.label_disc_real, 1.0)) + \
                                       0.5 * tf.reduce_mean(tf.squared_difference(self.label_disc_fake, 0.0))
            self.label_adv_gen_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.label_disc_fake, 1.0))

        with tf.variable_scope('TextureDiscriminator_loss'):
            self.texture_adv_disc_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.texture_disc_real, 1.0)) + \
                                         0.5 * tf.reduce_mean(tf.squared_difference(self.texture_disc_fake, 0.0))
            self.texture_adv_gen_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.texture_disc_fake, 1.0))

        # _______
        # Mask Discriminator loss:
        # this is a vanilla GAN: use CrossEntropy as loss
        # with tf.variable_scope('LabelDiscriminator_loss'):
        #     self.label_adv_disc_loss = - tf.reduce_mean(tf.log(self.label_disc_real) + tf.log(1. - self.label_disc_fake))
        #     self.label_adv_gen_loss = - tf.reduce_mean(tf.log(self.label_disc_fake))
        #
        # with tf.variable_scope('TextureDiscriminator_loss'):
        #     self.texture_adv_disc_loss = - tf.reduce_mean(tf.log(self.texture_disc_real) + tf.log(1. - self.texture_disc_fake))
        #     self.texture_adv_gen_loss = - tf.reduce_mean(tf.log(self.texture_disc_fake))

        # _______
        # L2 regularization loss:
        # with tf.variable_scope('L2_regularization_loss'):
        #     self.l2_reg_loss = idas_losses.l2_weights_regularization_loss()

        # - - - - - - - - - - - -

        # define weights for the cost contributes:
        w_kl = 0.1
        w_image_rec = 1.0
        w_z_rec = 1.0
        w_label_rec = 1.0
        w_texture_rec = 1.0
        w_adv = 10.0

        # define losses for unsupervised and discriminator prediction steps:
        self.unsup_loss = w_kl * self.kl_div_loss + \
                          w_image_rec * self.unsup_image_rec_loss + \
                          w_z_rec * self.unsup_z_regress_loss + \
                          w_adv * self.label_adv_gen_loss + \
                          w_adv * self.texture_adv_gen_loss  \
                          # w_label_rec * self.label_rec_loss + \
                          # w_texture_rec * self.texture_rec_loss

        self.discriminator_loss = w_adv * self.label_adv_disc_loss + \
                                  w_adv * self.texture_adv_disc_loss

        # add regularization:
        # self.unsup_loss += 0.01 * self.l2_reg_loss

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
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

        clip = False  # todo: try true
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op_unsup = _train_op_wrapper(self.unsup_loss, optimizer, clip)

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "LabelDiscriminator")
        disc_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TextureDiscriminator"))
        self.train_op_adv_disc = _train_op_wrapper(self.discriminator_loss, optimizer, clip, var_list=disc_vars)

        disc_op_list = []
        for i in range(self.disc_times):
            disc_op_list.append(self.train_op_adv_disc)

        with tf.control_dependencies([self.train_op_unsup]):
            self.global_train_op = tf.group(disc_op_list)

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        # todo: insert a metric later
        return

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        # Scalar summaries:
        with tf.name_scope('Total_loss'):
            tr_disc = tf.summary.scalar('train/disc_loss', self.discriminator_loss)
            tr_unsup = tf.summary.scalar('train/unsup_loss', self.unsup_loss)

        with tf.name_scope('Reconstruction'):
            tr_image_rec = tf.summary.scalar('train/image_rec_loss', self.unsup_image_rec_loss)
            tr_label_rec = tf.summary.scalar('train/label_rec_loss', self.label_rec_loss)
            tr_tex_rec = tf.summary.scalar('train/texture_rec_loss', self.texture_rec_loss)

        with tf.name_scope('KL_Divergence'):
            tr_kl = tf.summary.scalar('train/kl_loss', self.kl_div_loss)

        # Image summaries:
        with tf.name_scope('0_Input'):
            img_inp_us = tf.summary.image('input_unsup', self.unsup_input_data, max_outputs=2)
        with tf.name_scope('1_Reconstruction'):
            img_rec_us = tf.summary.image('unsup_rec', self.unsup_reconstruction, max_outputs=2)

        def get_slice(incoming, idx): # for soft and hard anatomy
            return tf.expand_dims(incoming[..., idx], -1)

        N = self.n_anatomical_masks
        with tf.name_scope('2_SoftAnatomy'):
            img_s_an_lst = [tf.summary.image('soft_{0}'.format(i), get_slice(self.disc_soft_anatomy, i), max_outputs=1)
                            for i in range(N)]
        with tf.name_scope('3_HardAnatomy'):
            img_h_an_lst = [tf.summary.image('hard_{0}'.format(i), get_slice(self.disc_hard_anatomy, i), max_outputs=1)
                            for i in range(N)]

        # _______________________________
        # merging all scalar summaries:
        train_scalar_summaries = [tr_disc, tr_unsup, tr_image_rec, tr_label_rec, tr_tex_rec, tr_kl]

        self.train_scalar_summary_op = tf.summary.merge(train_scalar_summaries)

        # _______________________________
        # merging all images summaries:
        valid_images_summaries = [img_inp_us, img_rec_us, img_s_an_lst, img_h_an_lst]

        self.valid_images_summary_op = tf.summary.merge(valid_images_summaries)

        # ---- #
        if self.tensorboard_verbose:
            _vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in _vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def _train_all_op(self, sess, writer, step):
        _, usl, dl, scalar_summaries = sess.run([self.global_train_op,
                                                 self.unsup_loss,
                                                 self.discriminator_loss,
                                                 self.train_scalar_summary_op],
                                                feed_dict={self.is_training: True})

        if random.randint(0, self.train_summaries_skip) == 0:
            writer.add_summary(scalar_summaries, global_step=step)

        return usl, dl

    def train_one_epoch(self, sess, iterator_init_list, writer, step, caller, seed):
        """ train the model for one epoch. """
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

        total_disc_loss = 0
        total_unsup_loss = 0
        n_batches = 0

        try:
            while True:
                self.progress_bar.monitor_progress()

                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                unsup_loss, disc_loss = self._train_all_op(sess, writer, step)
                total_disc_loss += disc_loss
                total_unsup_loss += unsup_loss
                step += 1

                n_batches += 1
                if (n_batches % self.skip_step) == 0:
                    print('\r  ...training over batch {1}: {0} batch_disc_loss = {2:.4f}\tbatch_unsup_loss = {3:.4f} {0}'
                          .format(' ' * 3, n_batches, disc_loss, unsup_loss), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the epoch. Compute statistics here:
            total_loss = total_disc_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            delta_t = time.time() - start_time

        # update global epoch counter:
        sess.run(self.increase_g_epoch)
        sess.run(self.update_g_train_step, feed_dict={'update_value:0': step})

        # detach progress bar and update last time of arrival:
        self.progress_bar.detach()
        self.progress_bar.update_lta(delta_t)

        print('\033[31m  TRAIN\033[0m:{0}{0} average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step

    def _eval_all_op(self, sess, writer, step):
        usl, dsl, unsup_im_summ= \
            sess.run([self.unsup_loss, self.discriminator_loss,
                      self.valid_images_summary_op],
                     feed_dict={self.is_training: False})
        writer.add_summary(unsup_im_summ, global_step=step)
        return usl, dsl

    def eval_once(self, sess, iterator_init_list, writer, step, caller):
        """ Eval the model once """
        start_time = time.time()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init)

        total_disc_loss = 0
        total_unsup_loss = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                unsup_loss, disc_loss = self._eval_all_op(sess, writer, step)
                total_disc_loss += disc_loss
                total_unsup_loss += unsup_loss
                step += 1

                n_batches += 1
                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the validation set. Compute statistics here:
            total_loss = total_disc_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            delta_t = time.time() - start_time
            pass

        # update global epoch counter:
        sess.run(self.update_g_valid_step, feed_dict={'update_value:0': step})

        print('\033[31m  VALIDATION\033[0m:  average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step, avg_loss

    def test_once(self, sess, disc_test_init, writer, step, caller):
        """ Test the model once """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(disc_test_init)

        total_disc_loss = 0
        total_unsup_loss = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                unsup_loss, disc_loss = self._eval_all_op(sess, writer, step)
                total_disc_loss += disc_loss
                total_unsup_loss += unsup_loss
                step += 1
                n_batches += 1

        except tf.errors.OutOfRangeError:
            # End of the test set. Compute statistics here:
            avg_loss = (total_unsup_loss + total_disc_loss) / n_batches
            delta_t = time.time() - start_time

            step += 1
            value = summary_pb2.Summary.Value(tag="y_TEST/test/loss_avg", simple_value=avg_loss)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)
            pass

        # update global epoch counter:
        sess.run(self.update_g_test_step, feed_dict={'update_value:0': step})

        print('\033[31m  TEST\033[0m:{0}{0} \033[1;33m average loss = {1:.4f}\033[0m on \033[1;33m{2}\033[0m batches '
              '{0} Took: {3:.3f} seconds'.format(' ' * 3, avg_loss, n_batches, delta_t))
        return step

    def test(self, image_data, texture_data, label_data):
        """ Test the model on input_data """
        if self.standardize:
            print('Remember to standardize your data!')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Returning: (soft anatomy, hard anatomy, texture_rec, texture_output, '
                      'label_rec, label_output, image_rec)')
                output = sess.run([self.disc_soft_anatomy, self.disc_hard_anatomy,
                                   self.texture_rec, self.texture_output,
                                   self.label_rec, self.label_output,
                                   self.unsup_reconstruction],
                                  feed_dict={self.unsup_input_data: image_data,
                                             self.disc_texture_data: texture_data,
                                             self.disc_image_data: image_data,
                                             self.disc_label_data: label_data,
                                             self.is_training: False})
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
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

            # to continue last training
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.last_checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:  # to train the main model from scratch
                # restore AE
                variables = tf.contrib.framework.get_variables_to_restore()
                variables_to_restore = [v for v in variables if
                                        v.name.split('/')[0] == 'texture_ae' or v.name.split('/')[0] == 'label_ae']
                ae_saver = tf.train.Saver(variables_to_restore)  # keep_checkpoint_every_n_hours=2
                ckpt_ae = tf.train.get_checkpoint_state(os.path.dirname(self.ae_checkpoint_dir + '/checkpoint'))
                if ckpt_ae and ckpt_ae.model_checkpoint_path:
                    ae_saver.restore(sess, ckpt_ae.model_checkpoint_path)

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
                ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
                print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - {1} : '.format(ep_str, self.run_id))
                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                global_ep = sess.run(self.g_epoch)
                self.callbacks_kwargs['es_loss'] = sess.run(self.best_val_loss)

                seed = global_ep

                # TRAINING ------------------------------------------
                iterator_init_list = [self.disc_train_init,
                                      self.unsup_train_init]
                t_step = self.train_one_epoch(sess, iterator_init_list, writer, t_step, caller, seed)

                # VALIDATION ------------------------------------------
                if global_ep >= 15 or not ((global_ep + 1) % 5):  # when to evaluate the model
                    iterator_init_list = [self.disc_valid_init,
                                          self.unsup_valid_init]
                    v_step, val_loss = self.eval_once(sess, iterator_init_list, writer, v_step, caller)
                    self.callbacks_kwargs['es_loss'] = val_loss
                    sess.run(self.update_best_val_loss, feed_dict={'best_val_loss_value:0': val_loss})

                if self.tensorboard_verbose and (global_ep % 20 == 0):
                    # writing summary for the weights:
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

            _ = self.test_once(sess, self.disc_test_init, writer, test_step, caller)

        writer.close()


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Model()
    model.build()
    model.train(n_epochs=2)
