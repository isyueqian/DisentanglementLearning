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

        self.x_validation = np.load(os.path.join(root_dir, folder + '/{0}/{0}_img.npy'.format('validation')))

        self.x_test = np.load(os.path.join(root_dir, folder + '/{0}/{0}_img.npy'.format('test')))

    @staticmethod
    def _data_augmentation_ops(x_train):
        """
        Data augmentation pipeline (to be applied on training samples)
        :param x_train: image
        :return:
        """
        angles = tf.random_uniform((1, 1), minval=-pi / 2, maxval=pi / 2)
        x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')

        translations = tf.random_uniform((1, 2), minval=-10, maxval=10)
        x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')

        # image distortions:
        x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        x_train = tf.cast(x_train, tf.float32)

        # add noise as regularizer
        std = 0.02  # data are standardized
        noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=std)
        x_train = x_train + noise

        return x_train

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

            _train_images = tf.constant(self.x_train, dtype=tf.float32)
            _valid_images = tf.constant(self.x_validation, dtype=tf.float32)
            _test_images = tf.constant(self.x_test, dtype=tf.float32)

            # _train_images = self.split_data_to_tensor(self.x_train)
            # _valid_images = self.split_data_to_tensor(self.x_validation)
            # _test_images = self.split_data_to_tensor(self.x_test)

            train_data = tf.data.Dataset.from_tensor_slices(_train_images)
            valid_data = tf.data.Dataset.from_tensor_slices(_valid_images)
            test_data = tf.data.Dataset.from_tensor_slices(_test_images)

            train_data = train_data.shuffle(buffer_size=len(self.x_train), seed=seed)

            if standardize:
                print("Data won't be standardized, as they already have been pre-processed.")

            if augment:
                train_data = train_data.map(lambda x: self._data_augmentation_ops(x),
                                            num_parallel_calls=num_threads)

            seed2 = seed + 1
            train_data = train_data.shuffle(buffer_size=len(self.x_train), seed=seed2)
            # need to shuffle twice

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
            _image_data = iterator.get_next()

            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for valid_data
            test_init = iterator.make_initializer(test_data)  # initializer for test_data

            with tf.name_scope('unsup_input_data'):
                input_data = tf.reshape(_image_data, shape=[-1, self.input_size[0], self.input_size[1], self.ch])
                input_data = tf.cast(input_data, tf.float32)

            with tf.name_scope('unsup_output_data'):
                output_data = tf.reshape(_image_data, shape=[-1, self.input_size[0], self.input_size[1], self.ch])
                output_data = tf.cast(output_data, tf.float32)

            return train_init, valid_init, test_init, input_data, output_data
