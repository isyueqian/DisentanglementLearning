import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import sys


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

        image_name = 'sup_' + folder + '.npy'
        label_name = 'sup_mask_' + folder + '.npy'

        image = np.load(os.path.join(file_path, image_name))
        label = np.load(os.path.join(file_path, label_name))

        assert len(image.shape) == 4 and len(label.shape) == 4, "Wrong dimension for preprocessed image or label!"
        print("Processing folder: ", folder)
        texture = center_crop_texture(folder, label, image, (16, 16))
        print("Texture shape for folder {0}: {1}".format(folder, texture.shape))


def center_crop_texture(folder, label, image, patch_size=(16, 16), check_number=None):
    """
    crop the texture part from the input image
    :param folder: one of train/validation/test
    :param label: input label, for center coordinate computing, expected input dim [slices, width, height, channels=1]
    :param image: input image, fro crop, expected input dim [slices, width, height, channels=1]
    :param patch_size:  crop size, expected dim [w_size, h_size]
    :param check_number: debug option
    :return:
    """

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

    texutre = np.zeros((label.shape[0], patch_size[0]*2, patch_size[1]*2, 1))

    for k in range(label.shape[0]):
        texutre[k, ...] = image[k, beginy[k]:endy[k], beginx[k]:endx[k], :]

        if check_number and k == check_number:
            plt.imshow(np.squeeze(image[k, ...]))
            plt.plot(center[k][1], center[k][0], 'om')
            plt.show()

            plt.imshow(np.squeeze(texutre[k, ...]))
            plt.show()

            sys.exit()

    np.save(os.path.join(root_path, folder, "texture_input_{}.npy".format(folder)), texutre)
    return texutre


def double_check(check_number=(2, 3, 4, 7)):
    folder = "train"
    file_path = os.path.join(root_path, folder, "texture_input_{}.npy".format(folder))
    texture = np.load(file_path)
    texture = np.squeeze(texture)
    for n in check_number:
        plt.imshow(texture[n, ...])
        plt.show()


if __name__ == '__main__':

    root_path = os.path.join("data", "acdc_data", "preprocessed")

    create_texture()
    # double_check()



