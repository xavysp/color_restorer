"""

"""

import os
import glob
import h5py
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

from PIL import Image
import scipy.misc
import scipy.ndimage
import numpy as np
# import cv2

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def read_dataset_h5(path):
    """
    Read .h5 file format data h5py <<.File>>
    :param path:file path of desired file
    :return: dataset -> contain images data for training;
    label -> contain  training label values (ground truth)
    """
    with h5py.File(path, 'r') as hf:
        n_variables = len(list(hf.keys()))
        # choice = True  # write
        if n_variables==3:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            test = np.array(hf.get('test'))
        elif n_variables==2:

            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            test=None
        elif n_variables == 1:
            data = np.array(hf.get('data'))
            label=None
            test=None
        else:
            data = None
            label = None
            test = None
            print("Error reading path: ",path)

        print(n_variables, " vars opened from: ", path)
        return data, label, test


def save_results_h5(savepath,data, label, test = None, result_name=None, label_name=None):
    if result_name==None or label_name==None:
        if np.any(test == None):

            with h5py.File(savepath, 'w') as hf:
                hf.create_dataset('data', data=data)
                hf.create_dataset('label', data=label)
                print("Data [", data.shape, "and label ", label.shape, "] saved in: ", savepath)
        else:
            with h5py.File(savepath, 'w') as hf:
                hf.create_dataset('data', data=data)
                hf.create_dataset('label', data=label)
                hf.create_dataset('test', data=test)
                print("Data [", data.shape, ", label ", label.shape, "and test ", test.shape,"] saved in: ", savepath)

    else:
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset(result_name, data=data)
            hf.create_dataset(label_name, data=label)
            print(result_name, "[", data.shape, " and ", label_name, label.shape, "] saved in: ", savepath)

def normalization_data_0255(data):
    """
    data normalization in 0 till 1 range
    :param data:
    :return:
    """
    ep=0.000001
    if not len(data.shape)==2:
        n_imgs = data.shape[0]
        # data = np.float32(data)
        if data.shape[-1]==3 and len(data.shape)==3:

            data = ((data - np.min(data)) * 255 / ((np.max(data) - np.min(data))+ep))
            # data = ((data - np.min(data)) * 254 / (np.max(data) - np.min(data)))+1

        elif data.shape[-1] == 4 and len(data.shape) == 3:
            N = data[:, :, -1]
            RGB = data[:, :, 0:3]
            print(np.max(N), " -- ", np.max(RGB), '---', N.shape)
            N = ((N - np.min(N)) * 255 / ((np.max(N) - np.min(N)) + ep))
            N = np.expand_dims(N,axis=-1)
            print(N.shape)
            RGB = ((RGB - np.min(RGB)) * 255 / ((np.max(RGB) - np.min(RGB)) + ep))
            data = np.concatenate([RGB,N],axis=2)
            print(data.shape)

        elif data.shape[-1]==3 and len(data.shape)==4:
            for i in range(n_imgs):
                # R = data[i,:,:,0]
                # G = data[i,:,:,1]
                # B = data[i,:,:,2]
                # N = data[i,:,:,3]
                # data[i,:,:,0]= ((R-np.min(R))*255/(np.max(R)-np.min(R)))
                # data[i, :, :, 1] = ((G - np.min(G)) * 255 / (np.max(G) - np.min(G)))
                # data[i, :, :, 2] = ((B - np.min(B)) * 255 / (np.max(B) - np.min(B)))
                # data[i, :, :, 3] = ((N - np.min(N)) * 255 / (np.max(N) - np.min(N)))
                img = data[i,...]
                data[i,:,:,:] = ((img - np.min(img)) * 255 / (np.max(img) - np.min(img)))
        # print("Data normalized with:", data.shape[-1], "channels")
        return data

    elif data.shape[-1]==3 and len(data.shape)==3:
        # R = data[:, :, 0]
        # G = data[:, :, 1]
        # B = data[:, :, 2]
        # data[:, :, 0] = ((R-np.min(R))*255/(np.max(R)-np.min(R)))
        # data[:, :, 1] = ((G - np.min(G)) * 255 / (np.max(G) - np.min(G)))
        # data[:, :, 2] = ((B - np.min(B)) * 255 / (np.max(B) - np.min(B)))
        data = ((data - np.min(data)) * 255 / (np.max(data) - np.min(data)))
        return data
    elif len(data.shape)==2:
        data = ((data-np.min(data))*255/(np.max(data)-np.min(data)))
        return data


def normalization_data_01(data):
    """
    data normalization in 0 till 1 range
    :param data:
    :return:
    """
    ep = 0.000001
    if not (len(data.shape)<=3):
        n_imgs = data.shape[0]
        data = np.float32(data)
        if data.shape[-1]==3:
            for i in range(n_imgs):
                # R = data[i,:,:,0]
                img = data[i,:,:,:]
                data[i,:,:,:] = ((img - np.min(img))*1)/(np.max(img)-np.min(img))

        elif data.shape[-1]==4:
            for i in range(n_imgs):
                img = data[i,:,:,:]
                data[i,:,:,:] = ((img - np.min(img))*1)/(np.max(img)-np.min(img))

        elif data.shape[-1]==3 and len(data.shape)==3:
            for i in range(n_imgs):
                # R = data[i,:,:,0]
                # G = data[i,:,:,1]
                # B = data[i,:,:,2]
                data = ((data-np.min(data))*1/(np.max(data)-np.min(data)))
                # data[:,:,0]= ((R-np.min(R))*1/(np.max(R)-np.min(R)))
                # data[:, :, 1] = ((G - np.min(G)) * 1 / (np.max(G) - np.min(G)))
                # data[:, :, 2] = ((B - np.min(B)) * 1 / (np.max(B) - np.min(B)))

        elif data.shape[-1]==2:
            for i in range(n_imgs):
                im = data[i,:,:,0]
                N = data[i,:,:,-1]
                data[i,:,:,0]= ((im-np.min(im))*1/(np.max(im)-np.min(im)))
                data[i, :, :, -1] = ((N - np.min(N)) * 1 / (np.max(N) - np.min(N)))
            del im, N

        elif data.shape[-1]==1 or len(data.shape)==3:
            if not len(data.shape)==3:
                for i in range(n_imgs):
                    img = data[i, :, :, 0]

                    data[i, :, :, 0] = ((img - np.min(img)) * 1 / (np.max(img) - np.min(img)))

                del img
            else:
                for i in range(n_imgs):
                    img = data[i, :, :]

                    data[i, :, :] = ((img - np.min(img)) * 1 / (np.max(img) - np.min(img)))
                print("the sahpe of data is only 3 instead of 4 ", data.shape)
                del img

        print("Data normalized with:", data.shape[-1], "channels")
        return data

    else:
        if data.shape[-1]>3: #  for rgb and Nir data
            N = data[:, :, -1]
            RGB = data[:, :, 0:3]
            # print(np.max(N), " -- ", np.max(RGB), '---', N.shape)
            N = ((N - np.min(N)) * 1 / ((np.max(N) - np.min(N)) + ep))
            N = np.expand_dims(N,axis=-1)
            # print(N.shape)
            RGB = ((RGB - np.min(RGB)) * 1 / ((np.max(RGB) - np.min(RGB)) + ep))
            data = np.concatenate([RGB,N],axis=2)
            # print(data.shape)
        else:
            data = ((data - np.min(data)) * 1 / (np.max(data) - np.min(data)))
        return data

def normalization_data_101(data):
    """
    data normalization in 0 till 1 range
    :param data:
    :return:
    """
    if not (len(data.shape)<=3):
        n_imgs = data.shape[0]
        data = np.float32(data)
        if data.shape[-1]==3:
            for i in range(n_imgs):
                # R = data[i,:,:,0]
                img = data[i,:,:,:]
                data[i,:,:,:] = (((img - np.min(img))*1)-1)/(np.max(img)-np.min(img))

        elif data.shape[-1]==4:
            for i in range(n_imgs):
                img = data[i,:,:,:]
                data[i, :, :, :] = (((img - np.min(img)) * 1) - 1) / (np.max(img) - np.min(img))

        elif data.shape[-1]==2:
            for i in range(n_imgs):
                im = data[i,:,:,0]
                N = data[i,:,:,-1]
                data[i,:,:,0]= ((im-np.min(im))*1/(np.max(im)-np.min(im)))
                data[i, :, :, -1] = ((N - np.min(N)) * 1 / (np.max(N) - np.min(N)))
            del im, N

        elif data.shape[-1]==1 or len(data.shape)==3:
            if not len(data.shape)==3:
                for i in range(n_imgs):
                    img = data[i, :, :, 0]

                    data[i, :, :, 0] = ((img - np.min(img)) * 1 / (np.max(img) - np.min(img)))

                del img
            else:
                for i in range(n_imgs):
                    img = data[i, :, :]

                    data[i, :, :] = ((img - np.min(img)) * 1 / (np.max(img) - np.min(img)))
                print("the shape of data is only 3 instead of 4 ", data.shape)
                del img

        print("Data normalized with:", data.shape[-1], "channels")
        return data

    else:
        data = (((data - np.min(data)) * 1)-1) / (np.max(data) - np.min(data))
        return data


def ssim_psnr(img_pred, img_lab):
    """
    The Mean squared Error for image similarity

    :param img_pred:
    :param img_lab:
    :return:
    """
    # print("Img_pred ",img_pred.shape)
    # print("Img_lab ", img_lab.shape)

    img_pred = normalization_data_01(img_pred)
    img_lab = normalization_data_01(img_lab)

    img_pred = normalization_data_0255(img_pred)
    img_lab = normalization_data_0255(img_lab)

    if len(img_pred.shape)==4 and not img_pred.shape[-1]==3:
        img_pred = img_pred[0,:,:,0]
        img_lab = img_lab[0,:,:,0]

        # err = np.sum((img_pred.astype("float") - img_lab.astype("float"))**2)
        # err /= float(img_pred[0]*img_pred.shape[1])

        err_mse = np.linalg.norm(img_pred-img_lab)
        # ssim
        err_ssim = ssim(img_lab, img_pred, data_range=img_pred.max()-img_pred.min())
        # for psnr

        err_psnr = psnr(img_pred, img_lab)
        return err_mse, err_ssim, err_psnr
    elif len(img_pred.shape)==2:
        err_mse = mse(img_pred,img_lab)  # mse function
        # ssim
        err_ssim = ssim(img_lab, img_pred, data_range=img_pred.max() - img_pred.min())
        err_psnr = psnr(img_pred, img_lab)
        return err_mse, err_ssim, err_psnr
    elif len(img_pred.shape)==4 and img_pred.shape[-1]==3:
        img_pred = img_pred[0, :, :,:]
        img_lab = img_lab[0, :, :, :]

        mse_R = mse(img_pred[:, :, 0], img_lab[:, :, 0])
        mse_G = mse(img_pred[:, :, 1], img_lab[:, :, 1])
        mse_B = mse(img_pred[:, :, 2], img_lab[:, :, 2])

        ssim_R = ssim(img_lab[:, :, 0], img_pred[:, :, 0], data_range=img_pred[:, :, 0].max() - img_pred[:, :, 0].min())
        ssim_G = ssim(img_lab[:, :, 1], img_pred[:, :, 1], data_range=img_pred[:, :, 1].max() - img_pred[:, :, 1].min())
        ssim_B = ssim(img_lab[:, :, 2], img_pred[:, :, 2], data_range=img_pred[:, :, 2].max() - img_pred[:, :, 2].min())
        psnr_R = psnr(img_pred[:, :, 0], img_lab[:, :, 0])
        psnr_G = psnr(img_pred[:, :, 1], img_lab[:, :, 1])
        psnr_B = psnr(img_pred[:, :, 2], img_lab[:, :, 2])
        return (mse_B + mse_G + mse_R) / 3, (ssim_B + ssim_G + ssim_R) / 3, (psnr_B + psnr_G + psnr_R) / 3


    elif len(img_pred.shape)==3 and img_pred.shape[-1]==3:
        mse_i = mse(img_pred, img_lab)

        ssim_R = ssim(img_lab[:,:,0], img_pred[:,:,0], data_range=img_pred[:,:,0].max()-img_pred[:,:,0].min())
        ssim_G = ssim(img_lab[:, :, 1], img_pred[:, :, 1], data_range=img_pred[:, :, 1].max() - img_pred[:, :, 1].min())
        ssim_B = ssim(img_lab[:, :, 2], img_pred[:, :, 2], data_range=img_pred[:, :, 2].max() - img_pred[:, :, 2].min())
        # ssim_i = ssim(img_lab, img_pred, data_range=img_pred.max() - img_pred.min())

        psnr_R, psnr_G, psnr_B= psnr(img_pred, img_lab)
        # (mse_B+mse_G+mse_R)/3, (ssim_B+ssim_G+ssim_R)/3,
        return mse_i, (ssim_B+ssim_G+ssim_R)/3, (psnr_B+psnr_G+psnr_R)/3

    else:

        print("please check again")
        return None, None


def mse(img_pred, img_lab):

    if len(img_pred.shape)== len(img_lab.shape):
        if len(img_pred.shape)==3 and img_pred.shape[-1]==3:
            mse = np.mean(np.power(img_lab-img_pred,2))
            return mse
        elif len(img_pred.shape)==2 and img_pred.shape[-1]>4:
            mse = np.mean(np.power(img_lab - img_pred, 2))
            return mse
        else:
            print("the image size is not as defined [h*w*c or h*w]  or the image channels are not 3 ")
    else:
        print("the shape of both images have to be equals")


def psnr(img_pred, img_lab):
    """

    :param mse: mean sqquare error
    :return:
    """
    if np.max(img_pred)<=1 and np.max(img_lab)<=1:
        img_lab = normalization_data_0255(img_lab)
        img_pred = normalization_data_0255(img_pred)
    # assert np.max(img_pred)>1
    # assert np.max(img_lab) > 1
    # print("max min ", np.max(img_pred), np.min(img_pred), np.max(img_lab), np.min(img_lab))

    mse_R = mse(img_pred[:, :, 0], img_lab[:, :, 0])
    mse_G = mse(img_pred[:, :, 1], img_lab[:, :, 1])
    mse_B = mse(img_pred[:, :, 2], img_lab[:, :, 2])

    if mse_R<= 0 or mse_G<=0 or mse_B<=0:
        return 100, 100, 100
    else:
        psnr_R = 20 * np.log10(np.max(img_pred[:, :, 0]) / np.sqrt(mse_R))
        psnr_G = 20 * np.log10(np.max(img_pred[:, :, 1]) / np.sqrt(mse_G))
        psnr_B = 20 * np.log10(np.max(img_pred[:, :, 2]) / np.sqrt(mse_B))
        return psnr_R, psnr_G, psnr_B

