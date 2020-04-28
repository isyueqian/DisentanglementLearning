import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
# from test import test_ID

test_ID = '100D_1rec'
test_number = 4

gt_img_path = "data/acdc_data/preprocessed/test/disc_test.npy"
gt_lab_path = "data/acdc_data/preprocessed/test/disc_mask_test.npy"
gt_tex_path = "data/acdc_data/preprocessed/test/texture_input_test.npy"

root_path = "tmp/{}/".format(test_ID)
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

# for image visualization
#
# for test_number in range(7):
#     plt.subplot(331)
#     plt.imshow(np.squeeze(gt_img[test_number, ...]))
#     plt.title("gt_img")
#     plt.subplot(332)
#     plt.imshow(np.squeeze(gt_lab[test_number, ...]))
#     plt.title("gt_lab")
#     plt.subplot(333)
#     plt.imshow(np.squeeze(gt_tex[test_number, ...]))
#     plt.title("gt_tex")
#     plt.subplot(334)
#     plt.imshow(np.squeeze(image_rec[test_number, ...]))
#     plt.title("image_rec")
#     plt.subplot(335)
#     plt.imshow(np.squeeze(label_rec[test_number, ...]))
#     plt.title("label_rec")
#     plt.subplot(336)
#     plt.imshow(np.squeeze(texture_rec[test_number, ...]))
#     plt.title("texture_rec")
#     plt.subplot(337)
#     plt.imshow(soft_anatomy[test_number, ..., 0])
#     plt.title("soft_anatomy")
#     # plt.subplot(337)
#     # plt.imshow(np.squeeze(hard_anatomy[test_number, ...]))
#     # plt.title("hard_anatomy")
#     plt.subplot(338)
#     plt.imshow(np.squeeze(label_output[test_number, ...]))
#     plt.title("label_output")
#     plt.subplot(339)
#     plt.imshow(np.squeeze(texture_output[test_number, ...]))
#     plt.title("texture_output")
#     plt.show()

# for anatomy factor visualization
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
plt.show()



# test_path = "data/acdc_data/train_sup/patient019"
# files = os.listdir(test_path)
# for file in files:
#     if "frame" in file and "nii" in file:
#         image = nib.load(os.path.join(test_path, file))
#         image_array = image.get_fdata()
#         print(file, image_array.shape)

"""
patient019_frame11.nii.gz (216, 256, 11)
patient019_frame11_gt.nii.gz (216, 256, 11)
patient019_frame01_gt.nii.gz (216, 256, 11)
patient019_frame01.nii.gz (216, 256, 11)
"""

# original_path = "data/acdc_data/training/patient003"
# files = os.listdir(original_path)
# for file in files:
#     if "nii.gz" in file:
#         image = nib.load(os.path.join(original_path, file))
#         image_array = image.get_fdata()
#         print(file, image_array.shape)

"""
DATA: 3d ED/ES with gt [width, height, slices], 4d full without gt [width, height, slices, time]

patient001_frame12.nii.gz (216, 256, 10) 
patient001_frame12_gt.nii.gz (216, 256, 10)
patient001_frame01.nii.gz (216, 256, 10)
patient001_frame01_gt.nii.gz (216, 256, 10)

patient001_4d.nii.gz (216, 256, 10, 30)
"""

# path = "data/acdc_data/preprocessed/test"
# files = os.listdir(path)
#
# for file in files:
#     array = np.load(os.path.join(path, file))
#     print(file)
#     print(array.shape)

"""
disc_mask_test.npy
(266, 128, 128, 4)
disc_test.npy
(266, 128, 128, 1)
sup_mask_test.npy
(266, 128, 128, 4)
sup_test.npy
(266, 128, 128, 1)
unsup_test.npy
(3730, 128, 128, 1)
"""


















