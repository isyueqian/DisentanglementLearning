"""
For the test the pre-trained model
"""
#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from model import Model
import numpy as np
import os
import shutil

test_ID = '100D_1rec'

root = 'data/acdc_data/preprocessed/test/'
filename = 'disc_test.npy'  # chose the file to test, can double check with disc ground truth
texture_filename = 'texture_input_test.npy'
label_filename = 'disc_mask_test.npy'

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    input_data = np.load(root + filename).astype(np.float32)
    input_texture = np.load(root + texture_filename).astype(np.float32)
    input_label = np.load(root + label_filename).astype(np.float32)

    soft_anatomy, hard_anatomy, texture_rec, texture_output, label_rec, label_output, image_rec = \
        model.test(input_data[:7], input_texture[:7], input_label[:7])

    print(soft_anatomy.shape)
    print(hard_anatomy.shape)
    print(texture_rec.shape)
    print(texture_output.shape)
    print(label_rec.shape)
    print(label_output.shape)
    print(image_rec.shape)

    root_path = "tmp/" + test_ID + '/'
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
        os.makedirs(root_path)
    else:
        os.makedirs(root_path)
    np.save(root_path + "soft_anatomy.npy", soft_anatomy)
    np.save(root_path + "hard_anatomy.npy", hard_anatomy)
    np.save(root_path + "texture_rec.npy", texture_rec)
    np.save(root_path + "texture_output.npy", texture_output)
    np.save(root_path + "label_rec.npy", label_rec)
    np.save(root_path + "label_output.npy", label_output)
    np.save(root_path + "image_rec.npy", image_rec)







