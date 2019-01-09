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
import cv2 as cv
import h5py

# from tensorlayer.layers import *

import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint

from utls import (h5_reader,
                   normalization_data_01,
                   normalization_data_0255,
                   mse, ssim_psnr,
                   h5_writer)
from utilities.data_manager import *

from models.cdent import net as CDNet
from models.endenet import net as ENDENet


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 192,"""The size of the images to process""")
tf.app.flags.DEFINE_string("model_name",'CDNet',"Choise one of [CDNet, ENDENet]")
tf.app.flags.DEFINE_string('model_state', 'test',"""training or testing [train, test, None]""")
tf.app.flags.DEFINE_integer('num_channels', 3,"""The number of channels in the images to process""")
tf.app.flags.DEFINE_integer('batch_size', 16,"""The size of the mini-batch default 32""")
tf.app.flags.DEFINE_integer('num_epochs', 3001,"""The number of iterations during the training""")
tf.app.flags.DEFINE_integer('n_test', 100,"""The size of the mini-batch""")
tf.app.flags.DEFINE_float('margin', 1.0,"""The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,"""The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_string('dataset_dir', '/opt/dataset', """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('dataset_name', 'ssmihd', """Dataset used by nir_cleaner choice [omsiv or ssomsi]""")
tf.app.flags.DEFINE_string('train_list', 'train_list.txt', """File which contian the training data""")
tf.app.flags.DEFINE_string('test_list', 'test_list.txt', """File which contain the testing data""")
tf.app.flags.DEFINE_string('gpu_id', '0',"""The default GPU id to use""")
tf.app.flags.DEFINE_bool('is_training', False,"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('prev_train_dir', 'checkpoints',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('optimizer', 'adam',"""training or testing [adam or momentum]""")
tf.app.flags.DEFINE_bool('is_image', False,"""training or testing [adam or momentum]""")
tf.app.flags.DEFINE_string('task', 'restorations',"""training or testing [restoration,superpixels,edges]""")
tf.app.flags.DEFINE_bool('use_nir', False,"""True for using nir in the test and False...""")
tf.app.flags.DEFINE_bool('use_all_data', True,""" True to use all data training and testing""")


pp = pprint.PrettyPrinter()
def save_batch_pred(args, Y_hat=None, Y_hat_name=None):
    """
    Given a tensor of images Y_hat, save_batch_pred
    save as h5 file each single image in a batch
    with the respective name in Y_hat_name
    :param Y_hat:
    :param Y_hat_name:
    :return:
    """
    Yhat_dir = os.path.join(FLAGS.dataset_dir,
                            os.path.join(FLAGS.dataset_name, FLAGS.task))
    Yhat_dir = os.path.join(Yhat_dir,'Yhat') if args.use_all_data else os.path.join(Yhat_dir,
                                                        os.path.join(FLAGS.model_state,'Yhat'))
    if not os.path.exists(Yhat_dir):
        os.makedirs(Yhat_dir)

    if args.batch_size>1:

        for i in range(len(Y_hat_name)):
            tmp_name = Y_hat_name[i]
            tmp_name = tmp_name.replace("RGBNC","RGBNP")
            h5_writer(savepath=os.path.join(Yhat_dir,tmp_name),data=np.squeeze(Y_hat[i,:,:,:]))
    else:
        tmp_name = Y_hat_name
        tmp_name = tmp_name.replace("RGBN", "RGBNP")
        h5_writer(savepath=os.path.join(Yhat_dir, tmp_name), data=np.squeeze(Y_hat))


if FLAGS.model_name =="CDNet" or FLAGS.model_name=="ENDENet":

    if not FLAGS.is_training:
        running_mode = 'test'
        pp.pprint(FLAGS.__flags)
        if FLAGS.dataset_name == "omsiv" and  FLAGS.is_image==False:
            # opening the test dataset
            test_list = test_data_loader(FLAGS)
            n_test = len(test_list)
            dataset_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, FLAGS.task))
            dataset_path = os.path.join(dataset_dir, os.path.join(FLAGS.model_state,'X'))
            data= h5_reader(os.path.join(dataset_path,test_list[0]))
            data = data[0]
            data = normalization_data_01(data)
            X = data[:, :, 0:3]
            print("X size: ", X.shape)
            del data
        elif (FLAGS.dataset_name == "ssmihd" or FLAGS.dataset_name == "SSMIHD") and  FLAGS.is_image==False:
            # opening the test dataset
            test_list = test_data_loader(FLAGS)
            n_test = len(test_list)
            dataset_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, FLAGS.task))
            dataset_path = os.path.join(dataset_dir,'X') if FLAGS.use_all_data else \
                os.path.join(dataset_dir,os.path.join(FLAGS.model_state,'X'))

            data = h5_reader(os.path.join(dataset_path, test_list[0]))
            data = data[0]
            data = normalization_data_01(data)
            X = data[:, :, 0:3]
            print("X size: ", X.shape)
            del data

        elif FLAGS.is_image=="True": # for testing a single image
            if FLAGS.dataset_name=='ssmihd':
                sample_data = 'dataset/RGBN_001.h5'
                list4test = os.listdir(FLAGS.dataset_dir)
                list4test.sort()
                data, label, test = h5_reader(sample_data)
                data = normalization_data_01(data)
            elif FLAGS.dataset_name=='omsiv':
                dataset_dir = os.path.join(FLAGS.dataset_dir,
                                           os.path.join(FLAGS.dataset_name, 'test'))
                dataset_path = os.path.join(dataset_dir, FLAGS.test_file)
                data, label = h5_reader(dataset_path)
                data = normalization_data_01(data)
                label = normalization_data_01(label)
                X = data[:, :, :, 0:3]
                Y = label[:, ...]
                print("Y size: ", Y.shape)
                print("X size: ", X.shape)
                del data, label

            X = data[:, :, 0:3]
            # Y = label[:, ...]
            X=np.expand_dims(X,axis=0)
            # Y=np.expand_dims(Y,axis=0)
            # print("Y size: ", Y.shape)
            print("X size: ", X.shape)
            del data, label
        else:

            print("this implementation just works with omsiv")

        # ----------- tensor size -------------#
        hl = [3, 32, 64, 32, 64, 32, 3]
        BATCH_SIZE = FLAGS.batch_size
        IM_WIDTH = X.shape[1]
        IM_HEIGHT = X.shape[0]
        IM_CHANNELS = FLAGS.num_channels
        TENSOR_SHAPE = (BATCH_SIZE,IM_HEIGHT,IM_WIDTH, IM_CHANNELS)

        RGBN = tf.placeholder(tf.float32, shape=TENSOR_SHAPE)

        if FLAGS.model_name=="CDNet":

            Y_hat, tmp = CDNet(hl, RGBN, reuse=False, train=False)  # for testing

        elif FLAGS.model_name=="ENDENet":
            Y_hat, tmp = ENDENet(hl, RGBN, reuse=False, train=False)  # for testing
        del tmp
        # -----------Restore Graph -------#

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)

        checkpoint_dir = "checkpoints/"+FLAGS.model_name
        if not os.path.exists(checkpoint_dir):
            print("Trained parameters DIR nor found")
            sys.exit()

        tl.files.load_ckpt(sess=sess, mode_name='{}_{}.ckpt'.format(FLAGS.model_name,FLAGS.dataset_name),
                           save_dir=checkpoint_dir)

        if FLAGS.batch_size==1:

            for i in range(n_test):
                X, X_names = get_testing_batch(FLAGS,test_list,
                                               current_indx = i)
                X=np.float32(X)
                pred = sess.run(Y_hat, {RGBN: X})
                pred=normalization_data_01(pred)
                print(pred.shape)
                save_batch_pred(FLAGS,Y_hat=pred, Y_hat_name=X_names)

            print("CDNet testing finished successfully")
            sys.exit()

        else:
            idx = np.arange(n_test)
            for i in range(0,n_test,FLAGS.batch_size):
                X, X_names = get_testing_batch(FLAGS,test_list,
                                               current_indx = i)
                X=np.float32(X)
                pred = sess.run(Y_hat, {RGBN: X})
                pred = normalization_data_01(pred)
                print(pred.shape)
                save_batch_pred(FLAGS,Y_hat=pred, Y_hat_name=X_names)

            print("CDNet testing finished successfully")
            sess.close()
            sys.exit()

else:
    print("There is something bad, we cannot find other NN model")
    sys.exit()