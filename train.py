"""

"""

__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"

import tensorflow as tf
import tensorflow.keras as tfk
import platform
import argparse

import time

from tqdm import tqdm
import pprint
import matplotlib.pyplot as plt

from utls import (h5_reader,
                   normalization_data_01,
                   normalization_data_0255,
                   mse, ssim_psnr,
                   h5_writer)
from models.cdent import CDENT
from utilities.data_manager import *
from dataset_manager import DataLoader



parser = argparse.ArgumentParser(description="CDNet arguments")

parser.add_argument('--image_size',type=int, default= 128,help="The size of the images to process") # 192
parser.add_argument('--dataset_name',type=str, default= 'OMSIV', help="Dataset used by nir_cleaner choice [omsiv or ssmihd]")
parser.add_argument("--model_name",type=str, default='CDNet',help="Choise one of [CDNet, ENDENet]")
parser.add_argument('--num_channels',type=int, default= 3,help="The number of channels in the images to process")
parser.add_argument('--batch_size',type=int, default= 8,help="The size of the mini-batch")

parser.add_argument('--num_epochs',type=int, default= 10,help="The number of iterations during the training")
parser.add_argument('--margin', type=float, default=1.0,help="The margin value for the loss function")
parser.add_argument('--lr', type=float, default=1e-4,help="The learning rate for the SGD optimization")
parser.add_argument('--weight_decay', type=float, default=0.0002, help="Set the weight decay")
parser.add_argument('--use_base_dir', type=bool, default=False, help="True when you are going to put the base directory of OMSIV dataset")
parser.add_argument('--dataset_dir', type=str, default='../../dataset')

parser.add_argument('--train_list', type=str, default='train_list.txt', help="File which contian the training data")
parser.add_argument('--test_list', type=str, default='test_list.txt', help="File which contain the testing data")
parser.add_argument('--gpu_id', type=str, default='0',help="The default GPU id to use")
parser.add_argument('--model_state', type=str, default='train',help="training or testing [train, test, None]")
parser.add_argument('--prev_train_dir', type=str, default='checkpoints',help="training or testing [True or False]")
parser.add_argument('--task',type=str, default= 'restorations',help="training or testing [restoration,superpixels,edges]")
parser.add_argument('--use_nir',type=bool, default=False,help="True for using the NIR channel")

arg = parser.parse_args()

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


def train():
    if arg.model_name.lower() =="cdnet" or arg.model_name.lower()=="endenet":
        # ***************data preparation *********************
        if arg.model_state.lower()=='train':


            # dataset preparation for training
            running_mode = 'train'
            data4training = DataLoader(
                data_name=arg.dataset_name,arg=arg)

            my_model = CDENT()

            loss_mse = tfk.losses.mean_squared_error
            accuracy = tf.image.ssim
            optimizer =tfk.optimizers.Adam(learning_rate=arg.lr,
                                           beta_1=0.5)
            # compile model
            my_model.compile(optimizer=optimizer, loss=loss_mse)
            #
            # my_model.fit_generator(
            #     generator=data4training,use_multiprocessing=True,
            #     workers=6,epochs=arg.num_epochs)
            my_model.fit(data4training, epochs=arg.num_epochs)

            print('oh oh ')

        else:
            print("this implementation just works with omsiv")

        # ******************* NN modeling ***********************

if __name__=="__main__":

    train()