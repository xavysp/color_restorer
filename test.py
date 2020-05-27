# -*- coding: utf-8 -*-
# from __future__ import division
"""
Test the CDNet and ENDENet on OMSIV dataset
"""
__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"


import numpy as np
import tensorflow.keras as tfk
import tensorflow as tf
from skimage.measure import compare_psnr,compare_ssim

import sys
import os
import cv2 as cv
import h5py
import platform
# from tensorlayer.layers import *

import matplotlib.pyplot as plt
import argparse
import pprint

from utls import (h5_reader,cv_imshow,
                   image_normalization,
                   mse, ssim_psnr,
                   h5_writer)
from utilities.data_manager import *
from dataset_manager import DataLoader
from models.cdent import CDENT
from models.endenet import net as ENDENet


data_dir = '/opt/dataset' if platform.system()=="Linux" else '../../dataset'
parser = argparse.ArgumentParser(description="CDNet arguments")

parser.add_argument('--img_width', type=int,default=580,help="Image width 580")
parser.add_argument('--img_height', type=int,default=320,help="Image height 320")
parser.add_argument("--model_name",type=str,default='CDNet',help="CDNet, ENDENet")
parser.add_argument('--model_state', type=str,default='test',help="[train, test, None]")
parser.add_argument('--num_channels', type=int, default=3,help="The number of channels in the images to process")
parser.add_argument('--batch_size', type=int, default=1,help="The size of the mini-batch default 32")
parser.add_argument('--num_epochs', type=int, default=3001,help="The number of iterations during the training")
parser.add_argument('--n_test', type=int, default=100,help="The size of the mini-batch")
parser.add_argument('--margin', type=float, default=1.0,help="The margin value for the loss function")
parser.add_argument('--lr', type=float, default=1e-4,help="The learning rate for the SGD optimization")
parser.add_argument('--dataset_dir', type=str, default=data_dir, help="The default path to the patches dataset")
parser.add_argument('--dataset_name', type=str, default='OMSIV', help="[omsiv or ssomsi]")
parser.add_argument('--train_list', type=str, default='train_list.txt', help="File which contian the training data")
parser.add_argument('--test_list', type=str, default='test_list.txt', help="File which contain the testing data")
parser.add_argument('--gpu_id', type=str, default='0',help="The default GPU id to use")
parser.add_argument('--is_training', type=bool, default=False,help="training or testing [True or False]")
parser.add_argument('--ckpt_dir', type=str, default='checkpoints',help="training or testing [True or False]")
parser.add_argument('--optimizer', type=str, default='adam',help="training or testing [adam or momentum]")
parser.add_argument('--is_image', type=bool, default=False,help="training or testing [adam or momentum]")
parser.add_argument('--task', type=str, default='restorations',help="training or testing [restoration,superpixels,edges]")
parser.add_argument('--use_nir', type=bool, default=False,help="True for using nir in the test and False...")
parser.add_argument('--use_all_data', type=bool, default=True,help=" True to use all data training and testing")

arg = parser.parse_args()


def save_batch_pred(args, Y_hat=None, Y_hat_name=None):
    """
    Given a tensor of images Y_hat, save_batch_pred
    save as h5 file each single image in a batch
    with the respective name in Y_hat_name
    :param Y_hat:
    :param Y_hat_name:
    :return:
    """
    Yhat_dir = os.path.join(arg.dataset_dir,
                            os.path.join(arg.dataset_name, arg.task))
    Yhat_dir = os.path.join(Yhat_dir,'Yhat') if args.use_all_data else os.path.join(Yhat_dir,
                                                        os.path.join(arg.model_state,'Yhat'))
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


if arg.model_name =="CDNet" or arg.model_name=="ENDENet":

    if not arg.is_training:
        data4testing = DataLoader(
            data_name=arg.dataset_name, arg=arg)

        model_dir = arg.model_name.lower() + '2' + arg.dataset_name
        res_dir = os.path.join('results',model_dir)
        os.makedirs(res_dir,exist_ok=True)
        ckpnt_dir = os.path.join(arg.ckpt_dir,model_dir)
        ckpnt_path = os.path.join(ckpnt_dir,'saved_weights.h5')

        my_model = CDENT()

        loss_mse = tfk.losses.mean_squared_error
        optimizer = tfk.optimizers.Adam(learning_rate=arg.lr)
        my_model.compile(optimizer=optimizer, loss=loss_mse, metrics='mse')
        input_shape = data4testing.input_shape
        my_model.build(input_shape=input_shape)
        my_model.load_weights(filepath=ckpnt_path)
        # preds = my_model.predict(data4testing)
        my_model.summary()
        # evaluation
        imgs_ssim = []
        imgs_psnr = []
        for i,(x,y) in enumerate(data4testing):
            p = my_model(x)
            # tmp_shape = data4testing.imgs_shape[i]
            tmp_name = data4testing.imgs_name[i]
            p = p.numpy()
            y = np.squeeze(y)
            p = image_normalization(np.squeeze(p),img_min=0., img_max=1.)
            tmp_ssim = compare_ssim(y,p,gaussian_weights=True, multichannel=True)
            tmp_psnr = compare_psnr(y,p)
            imgs_ssim.append(tmp_ssim)
            imgs_psnr.append(tmp_psnr)
            print(i,tmp_name)

        res_img = np.concatenate((y,p),axis=1)
        cv_imshow(img=np.uint8(image_normalization(res_img)),title='last pred image'+tmp_name)
        imgs_psnr = np.array(imgs_psnr)
        imgs_ssim = np.array(imgs_ssim)
        print('-------------------------------------------')
        print('Evaluation finished on: ',arg.dataset_name,'dataset')
        print('PSNR: ', imgs_psnr.mean())
        print('SSIM: ', imgs_ssim.mean())
        print('-------------------------------------------')