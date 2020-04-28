"""
For the test the pre-trained model auto-encoder
"""

from pretrain_model import AEModel
import numpy as np

root = 'data/acdc_data/preprocessed/test/'
texture_filename = 'texture_input_test.npy'  # chose the file to test
label_filename = 'sup_mask_test.npy'

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

    np.save("tmp/fine-tune-output/texture_rec.npy", texture_rec)
    np.save("tmp/fine-tune-output/label_rec.npy", label_rec)










