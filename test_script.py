import numpy as np
import os

path = "data/acdc_data/preprocessed/test"
files = os.listdir(path)

for file in files:
    array = np.load(os.path.join(path, file))
    print(file)
    print(array.shape)

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
