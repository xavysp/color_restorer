# -*- coding: utf-8 -*-
# from __future__ import division
"""
Test the CDNet and ENDENet on OMSIV dataset
"""
__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"


import numpy as np
import tensorflow as tf
import tensorlayer as tl
import sys
import os
import cv2

# from tensorlayer.layers import *

import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint

from utls import (read_dataset_h5,
                   normalization_data_01,
                   normalization_data_0255,
                   mse, ssim_psnr,
                   save_results_h5)
from models.cdent import net as CDNet
from models.endenet import net as ENDENet


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 192,"""The size of the images to process""")
tf.app.flags.DEFINE_string("model_name",'ENDENet',"Choise one of [CDNet, ENDENet]")
tf.app.flags.DEFINE_integer('num_channels', 3,"""The number of channels in the images to process""")
tf.app.flags.DEFINE_integer('batch_size', 128,"""The size of the mini-batch""")
tf.app.flags.DEFINE_integer('num_epochs', 3001,"""The number of iterations during the training""")
tf.app.flags.DEFINE_float('margin', 1.0,"""The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,"""The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_string('dataset_dir', '/opt/dataset', """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('dataset_name', 'omsiv', """Dataset used by nir_cleaner choice [omsiv or ssomsi]""")
tf.app.flags.DEFINE_string('train_file', 'OMSIV_train_192.h5', """dataset choice:[OMSIV_train_192.h5 or SSOMSI_train_192.h5]""")
tf.app.flags.DEFINE_string('test_file', 'OMSIV_test_192.h5', """dataset choice: [OMSIV_test_192.h5 /SSOMSI_test_192.h5]""")
tf.app.flags.DEFINE_string('gpu_id', '0',"""The default GPU id to use""")
tf.app.flags.DEFINE_string('is_training', 'False',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('prev_train_dir', 'checkpoints',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('optimizer', 'adam',"""training or testing [adam or momentum]""")
tf.app.flags.DEFINE_string('is_image', 'True',"""training or testing [adam or momentum]""")


pp = pprint.PrettyPrinter()

if FLAGS.model_name =="CDNet" or FLAGS.model_name=="ENDENet":

    if FLAGS.is_training:
        running_mode = 'test'
        pp.pprint(FLAGS.__flags)
        if FLAGS.dataset_name == "omsiv" and  FLAGS.is_image=="False":
            dataset_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, 'test'))
            dataset_path = os.path.join(dataset_dir, FLAGS.test_file)
            data, label = read_dataset_h5(dataset_path)
            data = normalization_data_01(data)
            label = normalization_data_01(label)
            X = data[:, :, :, 0:3]
            Y = label[:, ...]
            print("Y size: ", Y.shape)
            print("X size: ", X.shape)
            del data, label

        elif FLAGS.is_image=="True": # for testing a single image
            img_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, 'test'))
            dataset_path = os.path.join(img_dir, 'OMSIV_raw_img.h5')
            data, label = read_dataset_h5(dataset_path)

            data = normalization_data_01(data)
            label = normalization_data_01(label)
            X = data[:, :, 0:3]
            Y = label[:, ...]
            X=np.expand_dims(X,axis=0)
            Y=np.expand_dims(Y,axis=0)
            print("Y size: ", Y.shape)
            print("X size: ", X.shape)
            del data, label
        else:

            print("this implementation just works with omsiv")

        # ----------- tensor size -------------#
        hl = [3, 32, 64, 32, 64, 32, 3]
        BATCH_SIZE = X.shape[0]
        IM_WIDTH = X.shape[1]
        IM_HEIGHT = X.shape[2]
        IM_CHANNELS = FLAGS.num_channels
        TENSOR_SHAPE = (BATCH_SIZE, IM_WIDTH,IM_HEIGHT, IM_CHANNELS)

        RGBN = tf.placeholder(tf.float32, shape=TENSOR_SHAPE)

        if FLAGS.model_name=="CDNet":

            Y_hat, tmp = CDNet(hl, RGBN, reuse=False, train=False)  # for testing

        elif FLAGS.model_name=="ENDENet":
            Y_hat, tmp = ENDENet(hl, RGBN, reuse=False, train=False)  # for testing

        # -----------Restore Graph -------#

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)

        checkpoint_dir = "checkpoints/"+FLAGS.model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tl.files.load_ckpt(sess=sess, mode_name='params_{}.ckpt'.format('train'),
                           save_dir=checkpoint_dir)
        n = X.shape[0] // BATCH_SIZE
        for i in range(n):
            pred = sess.run(Y_hat,{RGBN:X})

        pred = normalization_data_01(pred)

        if FLAGS.is_image=="True":
            img = np.squeeze(pred)
            img = img**0.4040
            img = np.uint8(img*255)
            cv2.imshow('random_result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pred_path = '/opt/dataset/omsiv/result/'+FLAGS.model_name+'_resImg.h5'
            save_results_h5(pred_path,pred, Y,X)

        else:

            n = np.random.permutation(pred.shape[0])[7]
            img = pred[n,...]
            img = img**0.4040
            img = np.uint8(img*255)
            cv2.imshow('random_result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pred_path = '/opt/dataset/omsiv/result/'+FLAGS.model_name+'_res.h5'
            save_results_h5(pred_path,pred, Y,X)

