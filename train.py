"""

"""

__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"

import tensorflow as tf
import tensorlayer as tl
import numpy as np

import os
import sys
import time
import random
from tqdm import tqdm
import pprint
import matplotlib.pyplot as plt

from utls import (h5_reader,
                   normalization_data_01,
                   normalization_data_0255,
                   mse, ssim_psnr,
                   h5_writer)
from models.cdent import net as CDNet
from utilities.data_manager import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 128,"""The size of the images to process""") # 192
tf.app.flags.DEFINE_string('dataset_name', 'ssmihd', """Dataset used by nir_cleaner choice [omsiv or ssmihd]""")
tf.app.flags.DEFINE_string("model_name",'CDNet',"Choise one of [CDNet, ENDENet]")
tf.app.flags.DEFINE_integer('num_channels', 3,"""The number of channels in the images to process""")
if FLAGS.dataset_name=="omsiv" or FLAGS.dataset_name=="OMSIV":
    tf.app.flags.DEFINE_integer('batch_size', 32,"""The size of the mini-batch""")
elif FLAGS.dataset_name=="ssmihd" or FLAGS.dataset_name=="SSMIHD":
    tf.app.flags.DEFINE_integer('batch_size', 64, """The size of the mini-batch""")
else:
    tf.app.flags.DEFINE_integer('batch_size', None, """The size of the mini-batch""")
tf.app.flags.DEFINE_integer('num_epochs', 7001,"""The number of iterations during the training""")
tf.app.flags.DEFINE_float('margin', 1.0,"""The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,"""The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_float('weight_decay', 0.0002, """Set the weight decay""")
tf.app.flags.DEFINE_string('dataset_dir', '/opt/dataset', """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('train_list', 'train_list.txt', """File which contian the training data""")
tf.app.flags.DEFINE_string('test_list', 'test_list.txt', """File which contain the testing data""")
tf.app.flags.DEFINE_string('gpu_id', '0',"""The default GPU id to use""")
tf.app.flags.DEFINE_string('model_state', 'train',"""training or testing [train, test, None]""")
tf.app.flags.DEFINE_string('prev_train_dir', 'checkpoints',"""training or testing [True or False]""")
tf.app.flags.DEFINE_string('optimizer', 'adam',"""training or testing [adam, adamW or momentum]""")
tf.app.flags.DEFINE_string('task', 'restorations',"""training or testing [restoration,superpixels,edges]""")
tf.app.flags.DEFINE_bool('use_nir',False,"""True for using the NIR channel""")

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
    mse= tf.losses.mean_squared_error(y, y_hat) # tf.loss.mse(label, prediction)
    return mse

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
    if FLAGS.model_state=='Train' or FLAGS.model_state=='train':

         # dataset preparation for training
        running_mode = 'train'
        pp.pprint(FLAGS.__flags)

        if FLAGS.dataset_name=="omsiv" or FLAGS.dataset_name=="omsiv":
            print("Dataset files")
            dataset_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, FLAGS.task))
            data_list_path = os.path.join(dataset_dir, FLAGS.model_state)
            data_info = dataset_spliter(list_file=FLAGS.train_list,list_path=data_list_path, base_dir=True)

        elif FLAGS.dataset_name=="ssmihd" or FLAGS.dataset_name=="SSMIHD":
            print("Dataset files")
            dataset_dir = os.path.join(FLAGS.dataset_dir,
                                       os.path.join(FLAGS.dataset_name, FLAGS.task))
            data_list_path = os.path.join(dataset_dir, FLAGS.model_state)

            data_info = dataset_spliter(list_file=FLAGS.train_list, list_path=data_list_path, base_dir=True)

    else:
        print("this implementation just works with omsiv")

    # ******************* NN modeling ***********************

    if __name__=="__main__":

        BATCH_SIZE = FLAGS.batch_size
        IM_WIDTH = FLAGS.image_size
        IM_HEIGHT = FLAGS.image_size
        IM_CHANNELS = FLAGS.num_channels
        TENSOR_SHAPE = (BATCH_SIZE, IM_HEIGHT, IM_WIDTH, IM_CHANNELS)
        with tf.name_scope('inputs'):
            RGBN = tf.placeholder(tf.float32, shape=TENSOR_SHAPE, name='X')
            RGB = tf.placeholder(tf.float32, shape=TENSOR_SHAPE,name='Y')
        hl=[3, 32, 64, 32, 64, 32, 3]
        print("RGBN: ",RGBN)
        print("RGB: ", RGB)
        if FLAGS.model_name =='CDNet':
            Y_hat, tmp = CDNet(hl, RGBN, reuse=False,train=True) # training
            Y_hatv,tmp = CDNet(hl, RGBN,reuse=True,train=False) # for validation

        else:
            print("We do not find other models")
            sys.exit()

        with tf.name_scope("Loss"):
            loss = mse_loss(Y_hat, RGB)

        with tf.name_scope("Loss_v"):
            loss_val = mse_loss(Y_hatv, RGB)

        with tf.name_scope("Accuracy"):
            right_pred =tf.equal(tf.argmax(Y_hat,1), tf.argmax(RGB,1))  # .outputs
            accuracy = tf.reduce_mean(tf.cast(right_pred, tf.float32))

        with tf.name_scope("PNSR"):
            psnr = psnr_metric(Y_hat, RGB, maxi=255.0)

        with tf.name_scope("PNSR_v"):
            psnr_val = psnr_metric(Y_hat, RGB, maxi=255.0)

        with tf.name_scope("Learning_rate"):
            lr_var = tf.Variable(FLAGS.learning_rate,trainable=False)

        if FLAGS.optimizer =="adam":
            train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
        elif FLAGS.optimizer == "momentum":
            train_op = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                                  momentum=0.9).minimize(loss)
        elif FLAGS.optimizer == "adamw" or FLAGS.optimizer == "adamW":
            train_op= tf.contrib.opt.AdamWOptimizer(weight_decay = FLAGS.weight_decay,
                                                           learning_rate=FLAGS.learning_rate).minimize(loss)
        else:
            print("There were just two optimizer, please try again")
            sys.exit()
        # to visualize tensorflow summary graph
        tf.summary.scalar("Training_Loss", loss)
        tf.summary.scalar("Training_PSNR", psnr)
        tf.summary.scalar("Training_Accuracy", accuracy)
        tf.summary.scalar("Valid_Loss", loss_val)
        tf.summary.scalar("Valid_PSNR", psnr_val)
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=4)
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)

        logs_train_dir = "logs/"+FLAGS.model_name+'/train'
        logs_test_dir = "logs/"+FLAGS.model_name+'/test'
        if not os.path.exists(logs_train_dir):
            os.makedirs(logs_train_dir)
        if not os.path.exists(logs_test_dir):
            os.makedirs(logs_test_dir)
        summary_train = tf.summary.FileWriter(logs_train_dir, sess.graph)
        summary_test = tf.summary.FileWriter(logs_test_dir,sess.graph)

        checkpoint_dir = "checkpoints/"+FLAGS.model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # restoring previous trained data
        if tf.train.latest_checkpoint(checkpoint_dir)==None:
            global_step = 0
            ini_epoch = 0
        else:
            check_file= tf.train.latest_checkpoint(checkpoint_dir)
            global_step=int(''.join(list(filter(str.isdigit, check_file))))
            ini_epoch =global_step//len(data_info['train_indices'])
            tl.files.load_ckpt(sess=sess, mode_name='{}_{}.ckpt'.format(FLAGS.model_name,
                                                                        FLAGS.dataset_name),
                           save_dir=checkpoint_dir)
            # tl.files.load_ckpt(sess=sess, mode_name='{}_{}'.format(FLAGS.model_name,
            #                                                             FLAGS.dataset_name),
            #                    save_dir=checkpoint_dir)
        # training...
        def train_model(sess, dataset_info,batch_size, global_step,Train=False):
            ids1 = dataset_info['train_indices'].copy()
            np.random.shuffle(ids1)
            ids2 = dataset_info['train_indices'].copy()
            np.random.shuffle(ids2)
            dataset_paths = dataset_info['files_path']
            pbar = tqdm(range(len(ids1)))

            for step in pbar:
                start_time = time.time()
                batch_x, batch_y = data_parser(args=FLAGS,files_path=dataset_paths,
                                               indx1=ids1[step], indx2=ids2[step], batch_size=batch_size)
                feed_dict = {RGBN:batch_x, RGB:batch_y}
                _,loss_val, summary, y_hat = sess.run([train_op,loss, merged_summary_op, Y_hat],
                                                      feed_dict=feed_dict)
                global_step+=1
                summary_train.add_summary(summary, global_step=global_step)
            tmp_idx =np.random.permutation(batch_size)[0]
            duration = time.time() - start_time
            if global_step % 100 == 0:
                pbar.set_description('loss = %.7f (%.5f sec)' % (loss_val, duration))

            return loss_val, y_hat[tmp_idx,...],batch_y[tmp_idx,...], global_step

        def valid_model(sess, dataset_info,batch_size, global_step):
            ids1 = dataset_info['validation_indices'].copy()
            np.random.shuffle(ids1)
            ids2 = dataset_info['validation_indices'].copy()
            np.random.shuffle(ids2)
            dataset_paths = dataset_info['files_path']
            n = len(ids1)
            m_res = np.zeros((batch_size, 3))

            for i in range(n):
                x,y = data_parser(FLAGS,dataset_paths,indx1=ids1[i],indx2=ids2[i],batch_size=batch_size)
                feed_dict={RGBN:x,RGB:y}
                l_v,y_hat,summary_val = sess.run([loss_val,Y_hatv,merged_summary_op],feed_dict=feed_dict)
                summary_test.add_summary(summary_val, global_step=global_step)

            for i in range(x.shape[0]):
                m_res[i, 0], m_res[i, 1], m_res[i, 2] = ssim_psnr(np.float32(y_hat[i, ...]),np.float32(y[i, ...]))

            return l_v,m_res, np.float32(y_hat[27,...]), np.float32(y[27,...])

        initial_time= time.time()
        # ini_epoch =global_step//len(data_info['train_indices'])
        for epoch in range(ini_epoch,FLAGS.num_epochs):
            epoch_time = time.time()
            print("training...")
            l,y_hat,y,g_s=train_model(sess, data_info,BATCH_SIZE,global_step,True)

            y_hat = normalization_data_01(y_hat)
            y = normalization_data_01(y)
            tmp_im = np.concatenate((normalization_data_0255(y_hat**0.4040),
                                     normalization_data_0255(y**0.4040)))

            # train visualization
            plt.title("Epoch:"+str(epoch+1)+" Loss:"+'%.5f' % l+" training")
            plt.imshow(np.uint8(tmp_im))
            plt.draw()
            plt.pause(0.0001)

            global_step+= len(data_info['train_indices'])# n_train//BATCH_SIZE
            #saving...
            current_time = (time.time()-initial_time)/60
            if (epoch)%1000==0 or((current_time/60)>12):
                # if (current_time/60)>12:
                initial_time = time.time()
                tl.files.save_ckpt(sess=sess, mode_name='{}_{}.ckpt'.format(FLAGS.model_name,FLAGS.dataset_name),
                                   save_dir=checkpoint_dir, global_step=global_step)
                print("training saved in: ", epoch)
            #validation...
            if (epoch)%5==0:
                print("Validating...")
                l_val,metrics,y_hatv,y_val = valid_model(sess,data_info,BATCH_SIZE,global_step)
                y_hatv = normalization_data_01(y_hatv)
                y_val = normalization_data_01(y_val)
                tmp_im = np.concatenate((normalization_data_0255(y_hatv ** 0.4040),
                                         normalization_data_0255(y_val ** 0.4040)))

                # Valid visualization
                plt.title("Epoch:" + str(epoch + 1) + " Loss:" + '%.5f' % np.float32(l_val) + "Validating")
                plt.imshow(np.uint8(tmp_im))
                plt.draw()
                plt.pause(0.0001)
                print(
                    "Validation MSE-SSIM-PSNR: {} in {} epochs steps {}".format(np.float16(np.mean(metrics, axis=0)),
                                                                                epoch,
                                                                                global_step))
            print('Training loss = %.7f in %d epochs, %d steps.' % (l, epoch, global_step))



else:
    print("There is something bad, we cannot find other NN model")
    sys.exit()
