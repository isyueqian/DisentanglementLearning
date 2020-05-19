import tensorflow as tf
import numpy as np
import os
from math import pi
from idas.utils import print_yellow_text
# from idas.utils import get_available_gpus


class DatasetInterface(object):

    def __init__(self, root_dir, input_size, texture_size=32, ch=1):
        """
        Interface to the CUB data set for texture/image/label input
        :param root_dir: (string) path to directory containing CUB training data
        :param input_size: (int, int) input size for the neural network. It should be the same across all data sets
        :param texture_size: size of input texture
        :param ch: channels of the image and label
        """

        self.input_size = input_size  # to control the size of label and image
        self.texture_size = texture_size
        self.ch = ch

        folder = 'preprocessed_n_1'

        self.x_train = np.load(os.path.join(root_dir, folder + '/{0}/{0}_img.npy'.format('train')))
        self.y_train = np.load(os.path.join(root_dir, folder + '/{0}/{0}_gt.npy'.format('train')))
        self.t_train = np.load(os.path.join(root_dir, folder + '/{0}/texture_input_{0}.npy'.format('train')))

        self.x_validation = np.load(os.path.join(root_dir, folder + '/{0}/{0}_img.npy'.format('validation')))
        self.y_validation = np.load(os.path.join(root_dir, folder + '/{0}/{0}_gt.npy'.format('validation')))
        self.t_validation = np.load(os.path.join(root_dir, folder + '/{0}/texture_input_{0}.npy'.format('validation')))

        self.x_test = np.load(os.path.join(root_dir, folder + '/{0}/{0}_img.npy'.format('test')))
        self.y_test = np.load(os.path.join(root_dir, folder + '/{0}/{0}_gt.npy'.format('test')))
        self.t_test = np.load(os.path.join(root_dir, folder + '/{0}/texture_input_{0}.npy'.format('test')))

    @staticmethod
    def _data_augmentation_ops(x_train, y_train, t_train):
        """
        Data augmentation pipeline (to be applied on training samples)
        :param x_train: image
        :param t_train: texture
        :param y_train: label
        :return:
        """
        angles = tf.random_uniform((1, 1), minval=-pi / 2, maxval=pi / 2)
        x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')
        t_train = tf.contrib.image.rotate(t_train, angles[0], interpolation='BILINEAR')
        y_train = tf.contrib.image.rotate(y_train, angles[0], interpolation='NEAREST')

        translations = tf.random_uniform((1, 2), minval=-10, maxval=10)
        x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')
        t_train = tf.contrib.image.translate(t_train, translations, interpolation='BILINEAR')
        y_train = tf.contrib.image.translate(y_train, translations, interpolation='NEAREST')

        # image distortions:
        x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        x_train = tf.cast(x_train, tf.float32)
        t_train = tf.cast(t_train, tf.float32)
        y_train = tf.cast(y_train, tf.float32)

        # add noise as regularizer
        std = 0.02  # data are standardized
        noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=std)
        x_train = x_train + noise

        return x_train, y_train, t_train

    @staticmethod
    def split_data_to_tensor(data):
        size = data.shape[0]
        n = size // 2000
        tensor_list = []
        for i in range(n+1):
            if (i+1)*2000 < size:
                _temp_data = tf.constant(data[i*2000:(i+1)*2000], dtype=tf.float32)
            else:
                _temp_data = tf.constant(data[i*2000:], dtype=tf.float32)
            tensor_list.append(_temp_data)
        _data = tf.concat(tensor_list, axis=0)
        return _data

    def get_data(self, b_size, augment=False, standardize=False, repeat=False, num_threads=4, seed=None):
        """ Returns iterators on the dataset along with their initializers.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :param num_threads: for parallel computing
        :param seed: (int or placeholder) seed for the random operations,
                disc dataset could be shuffled twice with different seeds
        :return: train_init, valid_init, input_data, label
        """
        with tf.name_scope('cub_data'):

            # _train_images = tf.constant(self.x_train, dtype=tf.float32)
            # _train_masks = tf.constant(self.y_train, dtype=tf.float32)
            # _train_textures = tf.constant(self.t_train, dtype=tf.float32)
            #
            # _valid_images = tf.constant(self.x_validation, dtype=tf.float32)
            # _valid_masks = tf.constant(self.y_validation, dtype=tf.float32)
            # _valid_textures = tf.constant(self.t_validation, dtype=tf.float32)
            #
            # _test_images = tf.constant(self.x_test, dtype=tf.float32)
            # _test_masks = tf.constant(self.y_test, dtype=tf.float32)
            # _test_textures = tf.constant(self.t_test, dtype=tf.float32)

            _train_images = self.split_data_to_tensor(self.x_train)
            _train_masks = self.split_data_to_tensor(self.y_train)
            _train_textures = self.split_data_to_tensor(self.t_train)

            _valid_images = self.split_data_to_tensor(self.x_validation)
            _valid_masks = self.split_data_to_tensor(self.y_validation)
            _valid_textures = self.split_data_to_tensor(self.t_validation)

            _test_images = self.split_data_to_tensor(self.x_test)
            _test_masks = self.split_data_to_tensor(self.y_test)
            _test_textures = self.split_data_to_tensor(self.t_test)

            train_data = tf.data.Dataset.from_tensor_slices((_train_images, _train_masks, _train_textures))
            valid_data = tf.data.Dataset.from_tensor_slices((_valid_images, _valid_masks, _valid_textures))
            test_data = tf.data.Dataset.from_tensor_slices((_test_images, _test_masks, _test_textures))

            train_data = train_data.shuffle(buffer_size=len(self.x_train), seed=seed)

            if standardize:
                print("Data won't be standardized, as they already have been pre-processed.")

            if augment:
                train_data = train_data.map(lambda x, y, t: self._data_augmentation_ops(x, y, t),
                                            num_parallel_calls=num_threads)

            # seed2 = seed + 1
            # train_data = train_data.shuffle(buffer_size=len(self.x_train), seed=seed2)
            # now no need to shuffle twice

            if repeat:
                print_yellow_text(' --> Repeat the input indefinitely  = True', sep=False)
                train_data = train_data.repeat()  # Repeat the input indefinitely

            train_data = train_data.batch(b_size, drop_remainder=True)
            valid_data = valid_data.batch(b_size, drop_remainder=True)
            test_data = test_data.batch(b_size, drop_remainder=True)

            # if len(get_available_gpus()) > 0:
            #     # prefetch data to the GPU
            #     # train_data = train_data.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
            #     train_data = train_data.apply(tf.data.experimental.copy_to_device("/gpu:0")).prefetch(1)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            _image_data, _label_data, _texture_data = iterator.get_next()

            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for valid_data
            test_init = iterator.make_initializer(test_data)  # initializer for test_data

            with tf.name_scope('disc_image_data'):
                image_data = tf.reshape(_image_data, shape=[-1, self.input_size[0], self.input_size[1], self.ch])
                image_data = tf.cast(image_data, tf.float32)

            with tf.name_scope('disc_label_data'):
                label_data = tf.reshape(_label_data, shape=[-1, self.input_size[0], self.input_size[1], self.ch])
                label_data = tf.cast(label_data, tf.float32)

            with tf.name_scope('disc_texture_data'):
                texture_data = tf.reshape(_texture_data, shape=[-1, self.texture_size, self.texture_size, self.ch])

            return train_init, valid_init, test_init, image_data, label_data, texture_data
