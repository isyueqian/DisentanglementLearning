import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import config_file

date = '0519-100e-clip'
# test_ID = config_file.RUN_ID
test_ID = 'CUB-100D-nz64-3rec-reduce-metrics-clip-100e-7'
test_ae_ID = config_file.AE_RUN_ID
# folder = 'preprocessed'
folder = 'preprocessed_n_1'
save_folder = 'save2codes'

# CUB data path
gt_img_path = "data/CUB_200_2011/{}/test/test_img.npy".format(folder)
gt_lab_path = "data/CUB_200_2011/{}/test/test_gt.npy".format(folder)
gt_tex_path = "data/CUB_200_2011/{}/test/texture_input_test.npy".format(folder)

# ACDC data path
# gt_img_path = "data/acdc_data/preprocessed/test/disc_test.npy"
# gt_lab_path = "data/acdc_data/preprocessed/test/disc_mask_test.npy"
# gt_tex_path = "data/acdc_data/preprocessed/test/texture_input_test.npy"


# for image visualization
def to_rgb_2d(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)
    :param img: the array to convert [nx, ny, channels]
    :returns img: the rgb image [nx, ny, 3]
    """
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    channels = img.shape[-1]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[2]):
        st = img[:, :, k]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img


def test_pretrain_model():
    root_path = "{}/{}/".format(save_folder, test_ae_ID)

    texture_rec_path = root_path + "texture_rec.npy"
    label_rec_path = root_path + "label_rec.npy"

    gt_img = np.load(gt_img_path)
    gt_lab = np.load(gt_lab_path)
    gt_tex = np.load(gt_tex_path)

    texture_rec = np.load(texture_rec_path)
    label_rec = np.load(label_rec_path)

    for test_number in range(7):
        # test_number = 0
        plt.subplot(231)
        plt.imshow(np.squeeze(gt_img[test_number, ...]))
        plt.title("gt_img")
        plt.subplot(232)
        plt.imshow(np.squeeze(gt_lab[test_number, ...]))
        plt.title("gt_lab")
        plt.subplot(233)
        plt.imshow(np.squeeze(gt_tex[test_number, ...]/255))
        plt.title("gt_tex")

        plt.subplot(235)
        plt.imshow(np.squeeze(label_rec[test_number, ...])/255)
        plt.title("label_output_ae")

        # texture_rec = get_view(texture_rec)
        plt.subplot(236)
        plt.imshow((np.squeeze(texture_rec[test_number, ...])/255))
        plt.title("texture_output_ae")
        plt.savefig("image_results/{}_ae_output_{}.png".format(date, test_number))

        # plt.show()


def test_model():
    root_path = "{}/{}/".format(save_folder, test_ID)

    texture_rec_path = root_path + "texture_rec.npy"
    texture_output_path = root_path + "texture_output.npy"
    label_rec_path = root_path + "label_rec.npy"
    label_output_path = root_path + "label_output.npy"
    image_rec_path = root_path + "image_rec.npy"
    soft_anatomy_path = root_path + "soft_anatomy.npy"
    hard_anatomy_path = root_path + "hard_anatomy.npy"

    gt_img = np.load(gt_img_path)
    gt_lab = np.load(gt_lab_path)
    gt_tex = np.load(gt_tex_path)
    texture_rec = np.load(texture_rec_path)
    texture_output = np.load(texture_output_path)
    label_rec = np.load(label_rec_path)
    label_output = np.load(label_output_path)
    image_rec = np.load(image_rec_path)
    soft_anatomy = np.load(soft_anatomy_path)
    hard_anatomy = np.load(hard_anatomy_path)

    for test_number in range(7):
    # test_number = 2
        plt.subplot(331)
        plt.imshow(np.squeeze(gt_img[test_number, ...]))
        plt.title("gt_img")
        plt.subplot(332)
        plt.imshow(np.squeeze(gt_lab[test_number, ...]))
        plt.title("gt_lab")
        plt.subplot(333)
        plt.imshow(np.squeeze(gt_tex[test_number, ...])/255)
        plt.title("gt_tex")

        plt.subplot(334)
        plt.imshow(np.squeeze(image_rec[test_number, ...])/255)
        plt.title("image_rec")

        plt.subplot(335)
        plt.imshow(np.squeeze(label_rec[test_number, ...])/255)
        plt.title("label_rec")

        # texture_rec = get_view(texture_rec)
        plt.subplot(336)
        plt.imshow(np.squeeze(texture_rec[test_number, ...]/255))
        plt.title("texture_rec")

        # print(label_rec.shape)
        # print(np.max(label_rec))
        # print(np.min(label_rec))
        # print(np.max(texture_rec))
        # print(np.min(texture_rec))

        plt.subplot(337)
        plt.imshow(soft_anatomy[test_number, ..., 0])
        plt.title("soft_anatomy")
        # plt.subplot(337)
        # plt.imshow(np.squeeze(hard_anatomy[test_number, ...]))
        # plt.title("hard_anatomy")
        plt.subplot(338)
        plt.imshow(np.squeeze(label_output[test_number, ...])/255)
        plt.title("label_output")
        plt.subplot(339)
        plt.imshow(np.squeeze(texture_output[test_number, ...])/255)
        plt.title("texture_output")

        plt.savefig("image_results/{}_output_{}.png".format(date, test_number))

        plt.show()


def get_anatomy_factor():

    root_path = "{}/{}/".format(save_folder, test_ID)
    for test_number in range(7):

        soft_anatomy_path = root_path + "soft_anatomy.npy"
        hard_anatomy_path = root_path + "hard_anatomy.npy"

        gt_img = np.load(gt_img_path)
        gt_lab = np.load(gt_lab_path)
        gt_tex = np.load(gt_tex_path)

        soft_anatomy = np.load(soft_anatomy_path)
        hard_anatomy = np.load(hard_anatomy_path)
        big_soft = np.zeros([soft_anatomy.shape[1], 8*soft_anatomy.shape[2]])
        for i in range(8):
            print(soft_anatomy[test_number, ..., i].shape)
            big_soft[:, i*soft_anatomy.shape[2]:(i+1)*soft_anatomy.shape[2]] = soft_anatomy[test_number, ..., i]

        big_hard = np.zeros([hard_anatomy.shape[1], 8*hard_anatomy.shape[2]])
        for i in range(8):
            print(hard_anatomy[test_number, ..., i].shape)
            big_hard[:, i*hard_anatomy.shape[2]:(i+1)*hard_anatomy.shape[2]] = hard_anatomy[test_number, ..., i]

        plt.subplot(211)
        plt.imshow(big_soft)
        plt.title("soft_anatomy")
        plt.subplot(212)
        plt.imshow(big_hard)
        plt.title("hard_anatomy")
        plt.savefig("image_results/{}_anatomy_{}.png".format(date, test_number))


if __name__ == '__main__':

    # test_pretrain_model()

    # test_model()

    get_anatomy_factor()






















