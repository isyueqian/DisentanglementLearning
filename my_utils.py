import matplotlib.pyplot as plt
import os
import numpy as np
from idas.metrics.py_metrics import eval_dice, eval_miou
from idas.metrics.tf_metrics import generalized_dice_coe, miou_score
import tensorflow as tf


def binarization(label_rec):
    """
    input label_rec directly from the network, output binarized label_rec, for dice score computation
    :param label_rec:
    :return:
    """
    label_rec = label_rec / 255
    label_rec = tf.clip_by_value(label_rec, 0, 1)
    label_rec = tf.where(label_rec >= 0.5, tf.ones(label_rec.shape), tf.zeros(label_rec.shape))

    # debug option
    # for i in range(7):
    #     plt.imshow(np.squeeze(label_rec[i, ...]))
    #     plt.show()
    return label_rec


def get_score(label_rec, label_gt):
    """
    compute the dice score and mean IOU between reconstruction label and the ground truth label.
    :param label_rec: required size of [batch_size, height, width, channel], output map
    :param label_gt: required size of [batch_size, height, width, channel], original map
    :return:
    """

    binary_label_rec = binarization(label_rec)
    label_gt = tf.where(label_gt > 102, tf.ones(label_gt.shape), tf.zeros(
                        label_gt.shape))
    print(label_gt.shape)
    print(binary_label_rec.shape)

    # numpy implementation
    # dice = eval_dice(binary_label_rec, label_gt)
    # miou = eval_miou(binary_label_rec, label_gt)

    dice = generalized_dice_coe(binary_label_rec, label_gt)
    # dice = dice_coe(binary_label_rec, label_gt)
    miou = miou_score(binary_label_rec, label_gt)

    return dice, miou


def check_labels(label_gt, label_raw):

    label_binary = binarization(label_raw)

    for check_number in range(label_gt.shape[0]):
        print("check number: {}".format(check_number))
        plt.subplot(131)
        plt.imshow(np.squeeze(label_gt[check_number, ...]))
        plt.subplot(132)
        plt.imshow(np.squeeze(label_raw[check_number, ...]))
        plt.subplot(133)

        sess = tf.Session()
        with sess.as_default():
            label_binary = label_binary.eval()

        plt.imshow(np.squeeze(label_binary[check_number, ...]))
        # plt.savefig("{}.png".format(check_number))
        plt.show()


if __name__ == '__main__':
    test_path = "/home/yueqian/sdnet/tmp/CUB-100D-nz64-3rec-reduce"
    files = os.listdir(test_path)
    label_path = os.path.join(test_path, 'label_rec.npy')
    label_raw = np.load(label_path)

    folder = 'preprocessed_n_1'
    gt_lab_path = "data/CUB_200_2011/{}/test/test_gt.npy".format(folder)
    gt = np.load(gt_lab_path)

    dice, miou = get_score(label_raw, gt[:7])

    # sess = tf.Session()
    # with sess.as_default():
    #     print(dice.eval())
    #     print(miou.eval())
    #
    # print(dice, miou)

    check_labels(gt[:7], label_raw)












