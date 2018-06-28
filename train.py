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
from models.cdent import net as CDNet
from models.endenet import net as ENDENet


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 192,"""The size of the images to process""")
tf.app.flags.DEFINE_string("model_name",'CDNet',"Choise one of [CDNet, ENDENet]")
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
tf.app.flags.DEFINE_string('is_training', 'True',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('prev_train_dir', 'checkpoints',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('optimazer', 'adam',"""training or testing [adam or momentum]""")

pp = pprint.PrettyPrinter()

def mse_loss(y_hat, y):
    epsilon = 1e-12

    '''Y = tf.nn.l2_normalize(tar_tensor, dim=3)
    Y_hat = tf.nn.l2_normalize(pred_tensor, dim=3)
    loss = tf.losses.mean_squared_error(Y, Y_hat)'''
    y = ((y - tf.reduce_min(y, axis=[1, 2, 3], keepdims=True)) * 255) / ((tf.reduce_max(y,
        axis=[1, 2, 3],keepdims=True) - tf.reduce_min(y, axis=[1, 2, 3], keepdims=True))+epsilon)
    y_hat = ((y_hat- tf.reduce_min(y_hat, axis=[1, 2, 3], keepdims=True)) * 255) / ((
            tf.reduce_max(y_hat, axis=[1, 2, 3], keepdims=True) - tf.reduce_min(y_hat,axis=[1, 2, 3],
            keepdims=True))+epsilon)
    loss= tf.losses.mean_squared_error(y, y_hat) # tf.loss.mse(label, prediction)
    return loss

def psnr_metric(y_hat, y, maxi=255):
    epsilon = 0.0000001
    y = ((y - tf.reduce_min(y, axis=[1, 2, 3], keepdims=True)) * 255) / ((tf.reduce_max(y,
        axis=[1, 2, 3],keepdims=True) - tf.reduce_min(y, axis=[1, 2, 3], keepdims=True))+epsilon)
    y_hat = ((y_hat- tf.reduce_min(y_hat, axis=[1, 2, 3], keepdims=True)) * 255) / ((
            tf.reduce_max(y_hat, axis=[1, 2, 3], keepdims=True) - tf.reduce_min(y_hat,axis=[1, 2, 3],
            keepdims=True))+epsilon)

    mse = tf.losses.mean_squared_error(y_hat,y)  # (label, prediction)
    resu = tf.cond(mse<=0.0, lambda: 99.99, lambda: 10.0*tf.log(maxi**2/mse/tf.log(10.0)))

    return resu



if FLAGS.model_name =="CDNet" or FLAGS.model_name=="ENDENet":
    # ***************data preparation *********************
    if FLAGS.is_training:
         # dataset preparation for training
        pp.pprint(FLAGS.__flags)
        dataset_dir = os.path.join(FLAGS.dataset_dir,
                                   os.path.join(FLAGS.dataset_name,'train'))
        dataset_path = os.path.join(dataset_dir,FLAGS.train_file)
        data, label = read_dataset_h5(dataset_path)
        if FLAGS.dataset_name=="omsiv":
            data = normalization_data_01(data)
            label = normalization_data_01(label)
            X = data[:12371, :, :, 0:3]
            Xval = data[12371:-1, :, :,0:3]
            Y = label[:12371, ...]
            Yval = label[12371:-1, ...]
            print("Y size: ", Y.shape)
            print("Yval size: ", Yval.shape)
            print("X size: ", X.shape)
            print("Xval size: ",Xval.shape)
            del data, label
        else:
            print("this implementation just works with omsiv")

    else:
        # dataset preparation for testing
        pp.pprint(FLAGS.__flags)
        dataset_dir = os.path.join(FLAGS.dataset_dir,
                                   os.path.join(FLAGS.dataset_name, 'test'))
        dataset_path = os.path.join(dataset_dir, FLAGS.test_file)
        data, label = read_dataset_h5(dataset_path)
        if FLAGS.dataset_name == "omsiv":
            data = normalization_data_01(data)
            label = normalization_data_01(label)
            X = data[:, :, :, 0:3]
            Y = label[:, ...]
            print("Y size: ", Y.shape)
            print("X size: ", X.shape)
            del data, label
        else:
            print("this implementation just works with omsiv")

    # ******************* NN modeling ***********************

    if __name__=="__main__":
        with tf.name_scope('inputs'):
            BATCH_SIZE = FLAGS.batchsize
            IM_WIDTH= FLAGS.image_size
            IM_HEIGHT = FLAGS.image_size
            IM_CHANNELS = FLAGS.num_channels
            TENSOR_SHAPE= (BATCH_SIZE,IM_HEIGHT,IM_WIDTH,IM_CHANNELS)
            RGBN = tf.placeholder(tf.float32, shape=TENSOR_SHAPE, name='input')
            RGB = tf.placeholder(tf.float32, shape=TENSOR_SHAPE, name='label')
        hl=[3, 32, 64, 32, 64, 32, 3]

        if FLAGS.model_name =='CDNet':
            Y_hat, tmp = CDNet(hl, RGBN, reuse=False,train=True)
        elif FLAGS.model_name=='ENDENet':
            Y_hat, tmp = ENDENet(hl, RGBN, reuse=False,train=True)
        else:
            print("We do not find other models")
            sys.exit()
        with tf.name_scope("Loss"):
            loss = mse_loss(Y_hat, Y)

        with tf.name_scope("Accuracy"):
            right_pred =tf.equal(tf.argmax(Y_hat,1), tf.argmax(Y,1))  # .outputs
            accuracy = tf.reduce_mean(tf.cast(right_pred, tf.float32))

        with tf.name_scope("PNSR"):
            psnr = psnr_metric(Y_hat, Y, maxi=255.0)

        with tf.name_scope("Learning rate"):
            lr_var = tf.Variable(FLAGS.learngin_rate,trainable=False)

        if FLAGS.optimizer =="adam":
            train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learngin_rate).minimize(loss)
        elif FLAGS.optimizer == "momentum":
            train_op = tf.train.MomentumOptimizer(learning_rate=FLAGS.learngin_rate,
                                                  momentum=0.9).minimize(loss)
        else:
            print("There were just two optimizer, please try again")
            sys.exit()
        # to visualize tensorflow summary graph
        tf.summary.scalar("Loss", loss)
        tf.summary.scalar("PSNR", psnr)
        tf.summary.scalar("Accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)




else:
    print("There is something bad, we cannot find other NN model")
    sys.exit()
