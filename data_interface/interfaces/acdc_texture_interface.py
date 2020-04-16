import tensorflow as tf
import numpy as np
from idas.utils import get_available_gpus
import os
from math import pi
from idas.utils import print_yellow_text
import sys


class DatasetInterface(object):

    def __init__(self, root_dir, label_input_size, texture_input_size=(32, 32)):
        """
        Interface to the ADCD data set for input texture and labels
        :param root_dir: (string) path to directory containing ACDC training data
        :param label_input_size: (int, int) input size of label for the neural network. It should be the same across all data sets
        """
        self.label_input_size = label_input_size
        self.texture_input_size = texture_input_size

        self.texture_train = np.load(os.path.join(root_dir, 'preprocessed/{0}/texture_input_{0}.npy'.format('train')))
        self.label_train = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_mask_{0}.npy'.format('train')))

        self.texture_validation = np.load(os.path.join(root_dir, 'preprocessed/{0}/texture_input_{0}.npy'.format('validation')))
        self.label_validation = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_mask_{0}.npy'.format('validation')))

        self.texture_test = np.load(os.path.join(root_dir, 'preprocessed/{0}/texture_input_{0}.npy'.format('test')))
        self.label_test = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_mask_{0}.npy'.format('test')))


    @staticmethod
    def _data_augmentation_ops(x_train, y_train):
        """
        Data augmentation pipeline (to be applied on training samples)
        :param x_train: texture_input
        :param y_train: label_input
        :return:
        """
        angles = tf.random_uniform((1, 1), minval=-pi / 2, maxval=pi / 2)
        x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')
        y_train = tf.contrib.image.rotate(y_train, angles[0], interpolation='NEAREST')

        translations = tf.random_uniform((1, 2), minval=-10, maxval=10)
        x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')
        y_train = tf.contrib.image.translate(y_train, translations, interpolation='NEAREST')

        # image distortions:
        x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        x_train = tf.cast(x_train, tf.float32)
        y_train = tf.cast(y_train, tf.float32)

        # add noise as regularizer
        std = 0.02  # data are standardized
        noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=std)
        x_train = x_train + noise

        return x_train, y_train

    def get_data(self, b_size, augment=False, standardize=False, repeat=False, num_threads=4, seed=None):
        """
        Returns iterators on the dataset along with their initializers.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :param num_threads: for parallel computing
        :param seed: (int or placeholder) seed for the random operations
        :return: train_init, valid_init, test_init, input_texture, label
        """
        with tf.name_scope('acdc_data_pretrain'):

            _train_textures = tf.constant(self.texture_train, dtype=tf.float32)
            _train_labels = tf.constant(self.label_train, dtype=tf.float32)

            _valid_textures = tf.constant(self.texture_validation, dtype=tf.float32)
            _valid_labels = tf.constant(self.label_validation, dtype=tf.float32)

            _test_textures = tf.constant(self.texture_test, dtype=tf.float32)
            _test_labels = tf.constant(self.label_test, dtype=tf.float32)

            train_data = tf.data.Dataset.from_tensor_slices((_train_textures, _train_labels))
            valid_data = tf.data.Dataset.from_tensor_slices((_valid_textures, _valid_labels))
            test_data = tf.data.Dataset.from_tensor_slices((_test_textures, _test_labels))

            if standardize:
                print("Data won't be standardized, as they already have been pre-processed.")

            if augment:
                train_data = train_data.map(self._data_augmentation_ops, num_parallel_calls=num_threads)

            train_data = train_data.shuffle(buffer_size=len(self.texture_train), seed=seed)

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

            _texture_data, _label_data = iterator.get_next()
            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for valid_data
            test_init = iterator.make_initializer(test_data)

            with tf.name_scope('texture_input'):
                texture_data = tf.reshape(_texture_data, shape=[-1, self.texture_input_size[0], self.texture_input_size[1], 1])
                texture_data = tf.cast(texture_data, tf.float32)

            with tf.name_scope('output_sup'):
                label_data = tf.reshape(_label_data, shape=[-1, self.label_input_size[0], self.label_input_size[1], 1])
                label_data = tf.cast(label_data, tf.float32)

            return train_init, valid_init, test_init, texture_data, label_data
