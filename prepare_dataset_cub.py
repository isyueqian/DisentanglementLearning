"""
Running this file you will create pre-processed volumes to use to train the Network (useful to avoid to overload the CPU
during training).
In this way the pre-processing will be entirely off-line. Data augmentation is instead performed at
run time.

"""

import numpy as np
from glob import glob
from idas.utils import safe_mkdir, print_yellow_text
import os
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import sys
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# data set dirs:

source_dir = './data/CUB_200_2011/'
dest_dir = './data/CUB_200_2011/preprocessed_n_c/'  # with image label classes

safe_mkdir(dest_dir)

for subdir in ['train', 'validation', 'test']:
    safe_mkdir(os.path.join(dest_dir, subdir))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# input settings
mean_width = 467.89
mean_height = 386.03

# input image width and height
img_width = 384  # int(np.round(mean_height / 16) * 16)
img_height = 384  # int(np.round(mean_height / 16) * 16)

# image resolution
img_dx = 1.37
img_dy = 1.37

final_shape = (64, 64)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def resize_2d_slices(batch, new_size, interpolation):
    """
    Resize the frames
    :param batch: [np.array] input batch of images, with shape [n_batches, width, height, channels]
    :param new_size: [int, int] output size, with shape (N, M)
    :param interpolation: interpolation type
    :return: resized batch, with shape (n_batches, N, M, channels)
    """
    n_batches, x, y, ch = batch.shape
    output = []
    for k in range(n_batches):
        img = cv2.resize(batch[k], (new_size[1], new_size[0]), interpolation=interpolation)
        output.append(img)
    return np.array(output)


def crop_or_pad_slice_center(batch, new_size, value):
    """
    For every image in the batch, crop the image in the center so that it has a final size = new_size
    :param batch: [np.array] input batch of images, with shape [n_batches, width, height, channels]
    :param new_size: [int, int] output size, with shape (N, M)
    :return: cropped batch, with shape (n_batches, N, M, channels)
    """
    # pad always and then crop to the correct size:
    n_batches, x, y, ch = batch.shape

    pad_0 = (0, 0)
    pad_1 = (int(np.ceil(max(0, img_width - x)/2)), int(np.floor(max(0, img_width - x)/2)))
    pad_2 = (int(np.ceil(max(0, img_height - y)/2)), int(np.floor(max(0, img_height - y)/2)))

    if value is 'mean':
        batch = np.pad(batch, (pad_0, pad_1, pad_2, pad_0), mode='mean')
    elif value == 'min':
        batch = np.pad(batch, (pad_0, pad_1, pad_2, pad_0), mode='minimum')
    elif value == 'edge':
        batch = np.pad(batch, (pad_0, pad_1, pad_2, pad_0), mode='edge')
    else:
        c_value = value
        batch = np.pad(batch, (pad_0, pad_1, pad_2, pad_0), mode='constant', constant_values=c_value)

    # delta along axis and central coordinates
    n_batches, x, y, ch = batch.shape
    delta_x = new_size[0] // 2
    delta_y = new_size[1] // 2
    x0 = x // 2
    y0 = y // 2

    output = []
    for k in range(n_batches):
        output.append(batch[k,
                      x0 - delta_x: x0 + delta_x,
                      y0 - delta_y: y0 + delta_y, :])
    return np.array(output)


def clip_and_normalize(batch):
    """
    Standardize the input batch with z-score method.
    The batch images are also clipped to be in the interval 5th - 95th percentile.
    :param batch: (np.array) batch with images stacked on axis 0. Has shape [batch_size, width, height, channel].
    :return: transformed batch.
    """
    for i in range(batch.shape[-1]):

        lower_limit = np.percentile(batch[..., i], 5)
        upper_limit = np.percentile(batch[..., i], 95)
        batch[..., i] = np.clip(batch[..., i], a_min=lower_limit, a_max=upper_limit)

        m = np.mean(batch[..., i])
        s = np.std(batch[..., i])
        batch[..., i] = (batch[..., i] - m) / (s + 1e-12)

    assert not np.any(np.isnan(batch))

    return batch


def one_hot_encode(y, nb_classes):
    y_shape = list(y.shape)
    y_shape.append(nb_classes)
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            res = sess.run(tf.one_hot(indices=y, depth=nb_classes))
    return res.reshape(y_shape)


def image_pre_processing_pipline(filename):
    """
    Pre-processing pipeline for the input images
    :param filename: image file path
    :return:
    """
    # 1. load the .jpg file
    img = cv2.imread(filename)
    img = np.expand_dims(np.mean(img, axis=-1), axis=-1)

    # 2. create batch axis
    img_array = np.expand_dims(img, axis=0)

    # 3. crop to maximum size
    size = (img_width, img_height)
    img_array = crop_or_pad_slice_center(img_array, new_size=size, value='edge')

    # 4. undersample and make final size
    img_array = resize_2d_slices(img_array, new_size=final_shape, interpolation=cv2.INTER_CUBIC)
    img_array = np.expand_dims(img_array, axis=-1)

    # 5. standardize and clip values out of +- 3 standard deviations, will dramatically change the view
    img_array = clip_and_normalize(img_array)

    return img_array


def label_pre_processing_pipeline(filename, one_hot=False):
    """
    Pre-processing pipeline for input labels
    :param filename: label file path
    :param one_hot: if one-hot operation is needed
    :return:
    """
    # 1. load the .png file
    img = cv2.imread(filename)
    img = np.expand_dims(np.mean(img, axis=-1), axis=-1)

    # 2. create batch axis
    img_array = np.expand_dims(img, axis=0)

    # 3. crop to maximum size
    size = (img_width, img_height)
    img_array = crop_or_pad_slice_center(img_array, new_size=size, value=0)

    # 4. undersample and make final shape
    img_array = resize_2d_slices(img_array, new_size=final_shape, interpolation=cv2.INTER_NEAREST)
    img_array = np.expand_dims(img_array, axis=-1)

    # 5. one-hot encode:
    if one_hot:
        img_array = one_hot_encode(img_array, 6)  # level of gray

    return img_array


def build_data_set():

    # get train/val/test image_id lists
    train_test_split = pd.read_csv(os.path.join(source_dir, 'train_test_split.txt'),
                                   names=['image_id', 'is_training'],
                                   sep=' ')
    grouped = train_test_split.groupby(by='is_training')
    train_list = grouped.get_group(1)['image_id'].tolist()
    test_list = grouped.get_group(0)['image_id'].tolist()
    val_list = test_list[:len(test_list) // 2]
    test_list = test_list[len(test_list) // 2:]

    # get image data frame
    image_list_path = os.path.join(source_dir, 'images.txt')
    image_list = pd.read_csv(image_list_path, sep=' ', names=['image_name'], index_col=0)

    image_class_label_path = os.path.join(source_dir, 'image_class_labels.txt')
    image_class_list = pd.read_csv(image_class_label_path, sep=' ', names=['image_class'], index_col=0)

    # split image paths
    for i, op_list in enumerate([train_list, test_list, val_list]):
        if i == 0:
            kind = 'train'
        elif i == 1:
            kind = 'test'
        else:
            kind = 'validation'

        print("####### processing {} data #######".format(kind))
        img = []
        gt = []
        classes = []
        cnt = 0
        for i in op_list:
            cnt += 1
            print("Doing: {}/{}".format(cnt, len(op_list)))
            image_name = image_list.loc[i, 'image_name']
            image_class = image_class_list.loc[i, 'image_class']
            image_path = os.path.join(source_dir, 'images', image_name)
            label_path = os.path.join(source_dir, 'segmentations', image_name.replace('jpg', 'png'))

            image = image_pre_processing_pipline(image_path)
            label = label_pre_processing_pipeline(label_path, one_hot=False)

            # debug option
            # if cnt == 5:
            #     import sys
            #     print(image_class)
            #     print(type(image_class))
            #     plt.subplot(121)
            #     plt.imshow(np.squeeze(image))
            #     plt.subplot(122)
            #     plt.imshow(np.squeeze(label))
            #     plt.show()
            #     sys.exit()

            img.extend(image)
            gt.extend(label)
            classes.append(image_class)

        img_array = np.array(img)
        gt_array = np.array(gt)
        classes_array = np.array(classes)
        img_array, gt_array, classes_array = shuffle(img_array, gt_array, classes_array)

        np.save(os.path.join(dest_dir, '{0}/{0}_img.npy'.format(kind)), img_array)
        np.save(os.path.join(dest_dir, '{0}/{0}_gt.npy'.format(kind)), gt_array)
        np.save(os.path.join(dest_dir, '{0}/{0}_cls.npy'.format(kind)), classes_array)

        print(img_array.shape, gt_array.shape, classes_array.shape)


if __name__ == '__main__':
    build_data_set()




"""
[  0  51 102 153 204 255]
"""

