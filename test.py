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
import config_file

test_ID = config_file.RUN_ID
folder = 'preprocessed_n_1'
# folder = 'preprocessed'

root = 'data/CUB_200_2011/{}/test/'.format(folder)
filename = 'test_img.npy'  # chose the file to test, can double check with disc ground truth
texture_filename = 'texture_input_test.npy'
label_filename = 'test_gt.npy'

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    input_data = np.load(root + filename).astype(np.float32)
    input_texture = np.load(root + texture_filename).astype(np.float32)
    input_label = np.load(root + label_filename).astype(np.float32)

    num = input_data.shape[0]
    soft_anatomy_list = []
    hard_anatomy_list = []
    texture_rec_list = []
    texture_output_list = []
    label_rec_list = []
    label_output_list = []
    image_rec_list = []
    texture_code_list = []
    for i in range(num // 7):
        soft_anatomy, hard_anatomy, texture_rec, texture_output, label_rec, label_output, image_rec, texture_code = \
            model.test(input_data[7*i:7*(i+1)], input_texture[7*i:7*(i+1)], input_label[7*i:7*(i+1)])

        if i == 0:
            print(soft_anatomy.shape)
            print(hard_anatomy.shape)
            print(texture_rec.shape)
            print(texture_output.shape)
            print(label_rec.shape)
            print(label_output.shape)
            print(image_rec.shape)
            print(texture_code.shape)

        soft_anatomy_list.extend(soft_anatomy)
        hard_anatomy_list.extend(hard_anatomy)
        texture_rec_list.extend(texture_rec)
        texture_output_list.extend(texture_output)
        label_rec_list.extend(label_rec)
        label_output_list.extend(label_output)
        image_rec_list.extend(image_rec)
        texture_code_list.extend(texture_code)

    soft_anatomy_array = np.array(soft_anatomy_list)
    hard_anatomy_array = np.array(hard_anatomy_list)
    texture_rec_array = np.array(texture_rec_list)
    texture_output_array = np.array(texture_output_list)
    label_rec_array = np.array(label_rec_list)
    label_output_array = np.array(label_output_list)
    image_rec_array = np.array(image_rec_list)
    texture_code_array = np.array(texture_code_list)

    root_path = "save2codes/" + test_ID + '/'
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
        os.makedirs(root_path)
    else:
        os.makedirs(root_path)
    np.save(root_path + "soft_anatomy.npy", soft_anatomy_array)
    np.save(root_path + "hard_anatomy.npy", hard_anatomy_array)
    np.save(root_path + "texture_rec.npy", texture_rec_array)
    np.save(root_path + "texture_output.npy", texture_output_array)
    np.save(root_path + "label_rec.npy", label_rec_array)
    np.save(root_path + "label_output.npy", label_output_array)
    np.save(root_path + "image_rec.npy", image_rec_array)
    np.save(root_path + "texture_code.npy", texture_code_array)

    print(soft_anatomy_array.shape)
    print(hard_anatomy_array.shape)
    print(texture_rec_array.shape)
    print(texture_output_array.shape)
    print(label_rec_array.shape)
    print(label_output_array.shape)
    print(image_rec_array.shape)
    print(texture_code_array.shape)






