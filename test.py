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

root = 'data/acdc_data/preprocessed/test/'
filename = 'sup_test.npy'  # chose the file to test

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    input_data = np.load(root + filename).astype(np.float32)

    soft_anatomy, hard_anatomy, predicted_mask, reconstruction = model.test(input_data[:7])

    print(soft_anatomy.shape)
    print(hard_anatomy.shape)
    print(predicted_mask.shape)
    print(reconstruction.shape)

    np.save("tmp/soft_anatomy.npy", soft_anatomy)
    np.save("tmp/hard_anatomy.npy", hard_anatomy)
    np.save("tmp/predicted_mask.npy", predicted_mask)
    np.save("tmp/reconstruction.npy", reconstruction)







