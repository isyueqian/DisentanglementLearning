"""
For the test the pre-trained model auto-encoder
"""

from pretrain_model import AEModel
import numpy as np
import os
import config_file

# root = 'data/acdc_data/preprocessed/test/'
# texture_filename = 'texture_input_test.npy'  # chose the file to test
# label_filename = 'sup_mask_test.npy'

test_ID = config_file.AE_RUN_ID

root = 'data/CUB_200_2011/preprocessed_n_1/test/'
texture_filename = 'texture_input_test.npy'  # chose the file to test
label_filename = 'test_gt.npy'

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = AEModel()
    model.build()

    input_texture = np.load(root + texture_filename).astype(np.float32)
    input_label = np.load(root + label_filename).astype(np.float32)

    texture_rec, label_rec = model.test(input_texture[:7, ...], input_label[:7, ...])

    print(texture_rec.shape)
    print(label_rec.shape)

    save_folder = "tmp/{}/".format(test_ID)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_folder + "texture_rec.npy", texture_rec)
    np.save(save_folder + "label_rec.npy".format(test_ID), label_rec)










