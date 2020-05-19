import numpy as np
import matplotlib.pyplot as plt
import os

root_path = 'data/CUB_200_2011/preprocessed/test'

img_path = os.path.join(root_path, 'test_img.npy')
lab_path = os.path.join(root_path, 'test_gt.npy')
tex_path = os.path.join(root_path, 'texture_input_test.npy')

img = np.load(img_path)[:7]
lab = np.load(lab_path)[:7]
tex = np.load(tex_path)[:7]

print(img.shape)
print(lab.shape)
print(tex.shape)


for check in range(7):
    plt.subplot(131)
    plt.imshow(img[check])
    plt.subplot(132)
    plt.imshow(tex[check])
    plt.subplot(133)
    plt.imshow(lab[check])
    plt.savefig('test_input_{}.png'.format(check+1))
    plt.show()











