import tensorflow as tf

RUN_ID = 'CUB-100D-nz64-3rec-reduce-metrics-clip-100e'
AE_RUN_ID = 'CUB-AE-64-64-reduce'
# AE_RUN_ID = 'auto-encoder-new0420'

CUDA_VISIBLE_DEVICE = 2


def define_flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('RUN_ID', RUN_ID, "")
    tf.flags.DEFINE_string('AE_RUN_ID', AE_RUN_ID, "")

    # ____________________________________________________ #
    # ========== ARCHITECTURE HYPER-PARAMETERS ========== #

    # Learning rate:
    tf.flags.DEFINE_float('lr', 1e-4, 'learning rate')

    # batch size
    tf.flags.DEFINE_integer('b_size', 7, "batch size")

    tf.flags.DEFINE_integer('n_anatomical_masks', 8, "number of extracted anatomical masks")
    tf.flags.DEFINE_integer('n_frame_composing_masks', 8, "number composing masks for next frame mask prediction")
    tf.flags.DEFINE_integer('nz_latent', 64, "number latent variable for z code (encoder modality)")
    tf.flags.DEFINE_integer('n_ch_out', 1, "number of output channels - RGB or GRAY")
    # 1. config_file.py, cub_disc_interface.py, decoder.py, data_path
    tf.flags.DEFINE_integer('CUDA_VISIBLE_DEVICE', CUDA_VISIBLE_DEVICE, "visible gpu")
    tf.flags.DEFINE_integer('label_ae_filters', 64, "num of root filters of label auto-encoder")
    tf.flags.DEFINE_integer('texture_ae_filters', 64, "num of root filters of texture auto-encoder")
    tf.flags.DEFINE_integer('disc_times', 100, "num of training discriminators after training generator once")

    # ____________________________________________________ #
    # =============== TRAINING STRATEGY ================== #

    tf.flags.DEFINE_bool('augment', True, "Perform data augmentation")
    tf.flags.DEFINE_bool('standardize', False, "Perform data standardization (z-score)")  # data already pre-processed
    # (others, such as learning rate decay params...)

    # ____________________________________________________ #
    # =============== INTERNAL VARIABLES ================= #

    # internal variables:
    tf.flags.DEFINE_integer('num_threads', 20, "number of threads for loading data")
    tf.flags.DEFINE_integer('skip_step', 100, "frequency of printing batch report")
    tf.flags.DEFINE_integer('train_summaries_skip', 10, "number of skips before writing summaries for training steps "
                                                        "(used to reduce its verbosity; put 1 to avoid this)")
    tf.flags.DEFINE_bool('tensorboard_verbose', True, "if True: save also layers weights every N epochs")

    # ____________________________________________________ #
    # ===================== DATA SET ====================== #

    # ACDC data set:
    tf.flags.DEFINE_string('acdc_data_path', './data/acdc_data', "Path of ACDC data files.")
    tf.flags.DEFINE_string('cub_data_path', './data/CUB_200_2011', "Path of CUB data files")

    # data specs:
    tf.flags.DEFINE_list('input_size', [64, 64], "input size")  # 128
    tf.flags.DEFINE_integer('n_classes', 6, "number of classes")  # 4

    return FLAGS
