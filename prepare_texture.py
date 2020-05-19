import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

input_size = 64
texture_size = 16  # actually  window size = 32


def compute_center(label):
    """
    compute the center coordinate according to label
    :param label: expected input dim [slices, width, height, channels=1]
    :return:
    """
    points = np.where(label > 0)
    return np.array([[np.average(points[1][points[0] == j]), np.average(points[2][points[0] == j])]
                     for j in range(label.shape[0])])


def create_texture():
    """
    create texture for train/test/validation under the root path
    :return:
    """
    folders = os.listdir(root_path)
    for folder in folders:
        file_path = os.path.join(root_path, folder)

        if 'acdc' in data:
            image_name = 'disc_' + folder + '.npy'
            label_name = 'disc_mask_' + folder + '.npy'
        elif 'CUB' in data:
            image_name = folder + '_img.npy'
            label_name = folder + '_gt.npy'

        image = np.load(os.path.join(file_path, image_name))
        label = np.load(os.path.join(file_path, label_name))

        assert len(image.shape) == 4 and len(label.shape) == 4, "Wrong dimension for preprocessed image or label!"
        print("Processing folder: ", folder)
        texture = center_crop_texture(folder, label, image, (texture_size, texture_size))
        # texture = crop_to_shape(image, (32, 32))
        # np.save(os.path.join(file_path, "texture_input_{}.npy".format(folder)), texture)
        print("Texture shape for folder {0}: {1}".format(folder, texture.shape))


def center_crop_texture(folder, label, image, patch_size=(16, 16), check_number=None):
    """
    crop the texture part from the input image
    :param folder: one of train/validation/test
    :param label: input label, for center coordinate computing, expected input dim [slices, width, height, channels=1]
    :param image: input image, fro crop, expected input dim [slices, width, height, channels]
    :param patch_size:  1/2 crop size, expected dim [w_size, h_size]
    :param check_number: debug option
    :return:
    """
    ch = image.shape[-1]

    assert np.all(np.array(patch_size) <= np.array([label.shape[1], label.shape[2]])), print(
                    'Patch size exceeds dimensions.')

    center = compute_center(label)
    where_are_nan = np.isnan(center)
    center[where_are_nan] = int(label.shape[1] // 2)

    x = np.array([center[i][1] for i in range(label.shape[0])]).astype(np.int)
    y = np.array([center[i][0] for i in range(label.shape[0])]).astype(np.int)

    beginx = x - patch_size[0]
    beginy = y - patch_size[1]
    endx = x + patch_size[0]
    endy = y + patch_size[1]

    texture = np.zeros((label.shape[0], patch_size[0]*2, patch_size[1]*2, ch))

    for k in range(label.shape[0]):
        if beginx[k] >= 0 and endx[k] <= input_size and beginy[k] >= 0 and endy[k] <= input_size:
            texture[k, ...] = image[k, beginy[k]:endy[k], beginx[k]:endx[k], :]
        elif beginx[k] < 0 and endx[k] <= input_size and beginy[k] >= 0 and endy[k] <= input_size:
            texture[k, ...] = image[k, beginy[k]:endy[k], :patch_size[1]*2, :]
        elif beginx[k] >= 0 and endx[k] > input_size and beginy[k] >= 0 and endy[k] <= input_size:
            texture[k, ...] = image[k, beginy[k]:endy[k], input_size-2*patch_size[1]:, :]
        elif beginx[k] >= 0 and endx[k] <= input_size and beginy[k] < 0 and endy[k] <= input_size:
            texture[k, ...] = image[k, :patch_size[0]*2, beginx[k]:endx[k], :]
        elif beginx[k] >= 0 and endx[k] <= input_size and beginy[k] >= 0 and endy[k] > input_size:
            texture[k, ...] = image[k, input_size-2*patch_size[0]:, beginx[k]:endx[k], :]

        if check_number and k == check_number:
            print(np.max(texture[k, ...]))
            print(np.min(texture[k, ...]))
            print(texture[k, ...].shape)
            plt.subplot(121)
            plt.imshow(np.squeeze(image[k, ...]))
            plt.plot(center[k][1], center[k][0], 'om')
            plt.subplot(122)
            plt.imshow(np.squeeze(texture[k, ...])/255)
            plt.show()

            sys.exit()

    np.save(os.path.join(root_path, folder, "texture_input_{}.npy".format(folder)), texture)
    return texture


def double_check(check_number=(1, 5, 3, 6)):
    folder = "test"
    file_path = os.path.join(root_path, folder, "texture_input_{}.npy".format(folder))
    texture = np.load(file_path)
    texture = np.squeeze(texture)
    for n in check_number:
        plt.imshow(texture[n, ...]/255)
        plt.show()


def crop_to_shape(data, shape, mode='np'):
    """
    Crops the volumetric tensor or array into the given image shape by removing the border
    (expects a tensor or array of shape [n_batch, *vol_shape, channels]).

    :param data: the tensor or array to crop, shape=[n_batch, *vol_shape, n_class]
    :param shape: the target shape
    :param mode: 'np' or 'tf'.
    :return: The cropped tensor or array.
    """
    assert mode in ['np', 'tf'], "The mode must be either 'np' or 'tf'!"
    if mode == 'np':
        data_shape = data.shape
    elif mode == 'tf':
        data_shape = data.get_shape().as_list()
    else:
        raise NotImplementedError

    if len(shape) <= 3:
        shape = (1, ) + shape + (data_shape[-1], )

    assert np.all(tuple(data_shape[1:4]) >= shape[1:4]), "The shape of array to be cropped is smaller than the " \
                                                         "target shape."
    offset0 = (data_shape[1] - shape[1]) // 2
    offset1 = (data_shape[2] - shape[2]) // 2
    offset2 = (data_shape[3] - shape[3]) // 2
    remainder0 = (data_shape[1] - shape[1]) % 2
    remainder1 = (data_shape[2] - shape[2]) % 2
    remainder2 = (data_shape[3] - shape[3]) % 2

    if (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) == 0:
        return data

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, offset0:(-offset0 - remainder0), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, :, offset1:(-offset1 - remainder1), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, :, :, offset2:(-offset2 - remainder2), ]

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, :, offset1:(-offset1 - remainder1), offset2:(-offset2 - remainder2), ]

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, offset0:(-offset0 - remainder0), :, offset2:(-offset2 - remainder2), ]

    else:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1),
               offset2:(-offset2 - remainder2), ]


def get_roi_coordinates(label, mag_rate=0.1):
    """
    Produce the cuboid ROI coordinates representing the opposite vertices.

    :param label: A ground-truth label image.
    :param mag_rate: The magnification rate for ROI cropping.
    :return: An array representing the smallest coordinates of ROI;
        an array representing the largest coordinates of ROI.
    """
    label_intensity = np.unique(label)
    foreground_flag = np.any(np.concatenate(tuple([(np.expand_dims(label, axis=0) == k) for k in
                                                   label_intensity[1:]])), axis=0)
    arg_index = np.argwhere(foreground_flag)

    low = np.min(arg_index, axis=0)
    high = np.max(arg_index, axis=0)

    soft_low = np.maximum(np.floor(low - (high - low) * mag_rate / 2), np.zeros_like(low))
    soft_high = np.minimum(np.floor(high + (high - low) * mag_rate / 2), np.asarray(label.shape) - 1)

    return soft_low, soft_high


if __name__ == '__main__':

    # data = "acdc_data"
    data = "CUB_200_2011"
    root_path = os.path.join("data", data, "preprocessed_n_1")

    create_texture()
    # double_check()


    # test_path = "data/CUB_200_2011/preprocessed/train/train_img.npy"
    # img = np.load(test_path)[0:7]
    #
    # texture = crop_to_shape(img, (64, 64))
    # print(img.shape)
    # print(texture.shape)
    #
    # check_number = [1, 5, 3, 6]
    # for n in check_number:
    #     plt.imshow(texture[n, ...]/255)
    #     plt.show()


    # test_label = "data/CUB_200_2011/preprocessed/train/train_gt.npy"
    # label = np.load(test_label)[:7]
    # low, high = get_roi_coordinates(label)
    #
    # print(low)
    # print(high)








