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
tf.app.flags.DEFINE_integer('num_epochs', 5001,"""The number of iterations during the training""")
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
tf.app.flas.D

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
    if FLAGS.is_training:

         # dataset preparation for training
        running_mode = 'train'
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
        running_mode = 'test'
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
        elif FLAGS.model_name=='ENDENet':
            Y_hat, tmp = ENDENet(hl, RGBN, reuse=False,train=True)  # training
            Y_hatv, tmp = ENDENet(hl, RGBN, reuse=True, train=False)  # for validation
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

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)

        logs_train_dir = "logs/"+FLAGS.model_name+'/train'
        logs_test_dir = "logs"+FLAGS.model_name+'/test'
        if not os.path.exists(logs_train_dir):
            os.makedirs(logs_train_dir)
        if not os.path.exists(logs_test_dir):
            os.makedirs(logs_test_dir)
        summary_train = tf.summary.FileWriter(logs_train_dir, sess.graph)
        summary_test = tf.summary.FileWriter(logs_test_dir,sess.graph)

        checkpoint_dir = "checkpoints/"+FLAGS.model_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tl.files.load_ckpt(sess=sess, mode_name='params_{}.ckpt'.format(running_mode),
                           save_dir=checkpoint_dir)
        # training...
        def train_model(sess, x,y,n_train,batch_size, global_step,Train=False):
            idcs = np.random.permutation(range(n_train))
            pbar = tqdm(range(n_train // batch_size))

            for step in pbar:
                start_time = time.time()
                idx_i = idcs[step*batch_size:(step+1)*batch_size]
                batch_x = x[idx_i]
                batch_y = y[idx_i]
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

        def valid_model(sess, x,y,batch_size, global_step):
            n = x.shape[0]//batch_size
            m_res = np.zeros((x.shape[0], 3))
            for i in range(n):
                feed_dict={RGBN:x,RGB:y}
                l_v,y_hat,summary_val = sess.run([loss_val,Y_hatv,merged_summary_op],feed_dict=feed_dict)

            for i in range(x.shape[0]):
                m_res[i, 0], m_res[i, 1], m_res[i, 2] = ssim_psnr(y_hat[i, ...], y[i, ...])
            summary_test.add_summary(summary_val,global_step=global_step)
            return l_v,m_res, y_hat[27,...], y[27,...]

        global_step =0
        n_train = X.shape[0]
        initial_time= time.time()
        for epoch in range(FLAGS.num_epochs):
            epoch_time = time.time()
            print("training...")
            l,y_hat,y,g_s=train_model(sess, X,Y,n_train,BATCH_SIZE,global_step,True)

            y_hat = normalization_data_01(y_hat)
            y = normalization_data_01(y)
            tmp_im = np.concatenate((normalization_data_0255(y_hat**0.4040),
                                     normalization_data_0255(y**0.4040)))

            # train visualization
            plt.title("Epoch:"+str(epoch+1)+" Loss:"+'%.5f' % l+" training")
            plt.imshow(np.uint8(tmp_im))
            plt.draw()
            plt.pause(0.0001)

            global_step+= n_train//BATCH_SIZE
            #saving...
            current_time = (time.time()-initial_time)/60
            if (epoch)%1000==0 or((current_time/60)>12):
                # if (current_time/60)>12:
                initial_time = time.time()
                tl.files.save_ckpt(sess=sess, mode_name='params_{}.ckpt'.format(running_mode),
                                   save_dir=checkpoint_dir, global_step=global_step)
                print("training saved in: ", epoch)
            #validation...
            print("Validating...")
            l_val,metrics,y_hatv,y_val = valid_model(sess,Xval,Yval,BATCH_SIZE,global_step)
            y_hatv = normalization_data_01(y_hatv)
            y_val = normalization_data_01(y_val)
            tmp_im = np.concatenate((normalization_data_0255(y_hatv ** 0.4040),
                                     normalization_data_0255(y_val ** 0.4040)))

            # Valid visualization
            plt.title("Epoch:" + str(epoch + 1) + " Loss:" + '%.5f' % np.float32(l_val) + "Validating")
            plt.imshow(np.uint8(tmp_im))
            plt.draw()
            plt.pause(0.0001)

            print('Training loss = %.7f in %d epochs, %d steps.' % (l, epoch, global_step))
            print(
                "Validation MSE-SSIM-PSNR: {} in {} epochs steps {}".format(np.float16(np.mean(metrics, axis=0)), epoch,
                                                                            global_step))


else:
    print("There is something bad, we cannot find other NN model")
    sys.exit()
