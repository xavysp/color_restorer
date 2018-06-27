"""

"""
import tensorflow as tf
import numpy as np

import os
import sys
import time
import random
from tqdm import tqdm
import pprint
import matplotlib.pyplot as plt

from utls import (read_dataset_h5,
                   normalization_data_01,
                   normalization_data_0255,
                   mse, ssim_psnr,
                   save_variable_h5,
                   save_results_h5)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 192,"""The size of the images to process""")
tf.app.flags.DEFINE_string("training_model",'DAE',"Choise one of [DAE, triplets, CDED]")
tf.app.flags.DEFINE_integer('num_channels', 1,"""The number of channels in the images to process""")
tf.app.flags.DEFINE_integer('batch_size', 128,"""The size of the mini-batch""")
tf.app.flags.DEFINE_integer('num_epochs', 3001,"""The number of iterations during the training""")
tf.app.flags.DEFINE_float('margin', 1.0,"""The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,"""The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_string('dataset_dir', '/opt/dataset', """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('dataset_name', 'omsiv', """Dataset used by nir_cleaner choice [omsiv or ssomsi]""")
tf.app.flags.DEFINE_string('train_file', 'OMSIV_train_192.h5', """dataset choice:[OMSIV_train_192.h5 or SSOMSI_train_192.h5]""")
tf.app.flags.DEFINE_string('test_file', 'OMSIV_test_192.h5', """dataset choice: [OMSIV_test_192.h5 /SSOMSI_test_192.h5]""")
tf.app.flags.DEFINE_string('gpu_id', '0',"""The default GPU id to use""")
tf.app.flags.DEFINE_string('is_training', 'True',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('prev_train_dir', 'checkpoints',"""training or testing [True or False]""")

pp = pprint.PrettyPrinter()


if FLAGS.training_model =="CDNet" or FLAGS.training_model=="ENDENet":
    if FLAGS.is_training:
        pp.pprint(FLAGS.__flags)
        # dataset preparation
        dataset_dir = os.path.join(FLAGS.dataset_dir,
                                   os.path.join(FLAGS.dataset_name,'train'))
        dataset_name = os.path.join(dataset_dir,FLAGS.dataset_name)



    else:
        pass
else:
    print("There is something bad, we cannot find other NN model")
    sys.exit()
