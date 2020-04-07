import numpy as np
import os
import nibabel as nib

test_path = "data/acdc_data/train_sup/patient019"
files = os.listdir(test_path)
for file in files:
    if "frame" in file and "nii" in file:
        image = nib.load(os.path.join(test_path, file))
        image_array = image.get_fdata()
        print(file, image_array.shape)

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


















