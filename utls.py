"""

"""

import os, time
# import glob
import h5py
from termcolor import colored
# import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

import numpy as np
import cv2

def cv_imshow(img=None,title="on_title"):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def h5_reader(path):
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

        # print(n_variables, " vars opened from: ", path)
        return data, label, test


def h5_writer(savepath=None,data=None, label=None, test = None, result_name=None, label_name=None,n_val=1):

    if n_val==3 and (result_name==None or label_name==None):
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=data)
            hf.create_dataset('label', data=label)
            hf.create_dataset('test', data=test)
            print("Data [", data.shape, ", label ", label.shape, "and test ", test.shape, "] saved in: ", savepath)
    elif n_val==2 and (result_name==None or label_name==None):
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=data)
            hf.create_dataset('label', data=label)
            print("Data [", data.shape, "and label ", label.shape, "] saved in: ", savepath)
    elif n_val==1 and (result_name==None or label_name==None):
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('data', data=data, dtype='float32')
            print("Data [", data.shape, "] saved in: ", savepath)
    elif n_val==2 and (result_name is not None and label_name is not None):
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset(result_name, data=data)
            hf.create_dataset(label_name, data=label)
            print(result_name, "[", data.shape, " and ", label_name, label.shape, "] saved in: ", savepath)
    else:
        print('Sorry there is an error, please check our h5_writer() function')
        return

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
    epsilon = 1e-12
    if np.sum(np.isnan(data))>0:
        print('NaN detected before Normalization')
        return 'variable has NaN values'
    if len(data.shape)>3:
        # data with batch (tensors)
        n_imgs = data.shape[0]
        data = np.float32(data)
        if data.shape[-1]==3:
            for i in range(n_imgs):
                img = data[i,:,:,:]
                data[i,:,:,:] = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img))+epsilon))

        elif data.shape[-1]==4:
            print('it is a  little naive, check it in line 64 seg utils.py')
            for i in range(n_imgs):
                nir = data[i,:,:,-1]
                nir= ((nir - np.min(nir)) * 1 / ((np.max(nir) - np.min(nir))+epsilon))
                img = data[i,:,:,0:3]
                img = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img))+epsilon))
                data[i,:,:,0:3] = img
                data[i,:,:,-1]=nir
        elif data.shape[-1]==2:
            #normalization according to channels
            print('check line 70 utils_seg.py')
            for i in range(n_imgs):
                im = data[i,:,:,0]
                N = data[i,:,:,-1]
                data[i,:,:,0]= ((im-np.min(im))*1/(np.max(im)-np.min(im)))
                data[i, :, :, -1] = ((N - np.min(N)) * 1 / (np.max(N) - np.min(N)))
            del im, N

        elif data.shape[-1]==1:
            for i in range(n_imgs):
                img = data[i, :, :, 0]
                data[i, :, :, 0] = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img))+epsilon))
        else:
            print("error normalizing line 83")
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected after normalization')
            return 'variable has NaN values'
        return data

    else:
        # for single image (and [RGB,NIR] 4 channel)
        if np.max(data) ==0 and np.min(data)==0:
            return data
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected before normalization')
            return 'variable has NaN values'
        if len(data.shape)==3 and data.shape[-1]==4:
            nir = data[:, :, -1]
            nir = ((nir - np.min(nir)) * 1 / ((np.max(nir) - np.min(nir)) + epsilon))
            img = data[:, :, 0:3]
            img = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img)) + epsilon))
            data[:, :, 0:3] = img
            data[:, :, -1] = nir

        elif len(data.shape)==3 and data.shape[-1]>4:
            print('errro not implemented yet')
            return
        else:
            data = ((data - np.min(data)) * 1 / ((np.max(data) - np.min(data))+epsilon))
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected after normalization')
            return 'variable has NaN values'
        return data


def image_normalization(img, img_min=0, img_max=255):
    """ Image normalization given a minimum and maximum
    This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image given a scale
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img


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

def read_files_list(list_path,dataset_name=None):
    mfiles = open(list_path)
    file_names = mfiles.readlines()
    mfiles.close()

    file_names = [f.strip() for f in file_names]
    return file_names

def print_info(info_string, quite=False):

    info = '[{0}][INFO]{1}'.format(time.time(), info_string)
    print(colored(info, 'green'))

def print_error(error_string):

    error = '[{0}][ERROR] {1}'.format(time.time(), error_string)
    print (colored(error, 'red'))

def print_warning(warning_string):

    warning = '[{0}][WARNING] {1}'.format(time.time(), warning_string)

    print (colored(warning, 'blue'))
def img_post_processing(img):

    # Adjust Image intensity [0-255]

    width = img.shape[1]
    height = img.shape[0]
    R = img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    R= imadjust(R)
    G= imadjust(G)
    B= imadjust(B)
    # ***White balance***
    rgb_med = [np.mean(R), np.mean(G), np.mean(B)]
    rgb_scale = np.max(rgb_med)/rgb_med
    # Scale each color channel, to have the same median.
    R = R*rgb_scale[0]
    G = G*rgb_scale[1]
    B = B * rgb_scale[2]

    # ***restore bayer mosaic BGGR***
    I =np.zeros((height*2,width*2))
    I[0:height*2:2, 0:width*2:2] = B
    I[0:height* 2:2, 1:width* 2:2] = G
    I[1:height* 2:2, 1:width* 2:2] = R
    # image interpolation
    T = cv2.resize(G, (2*width,2*height),interpolation=cv2.INTER_CUBIC)
    I[1:height * 2:2, 0:width * 2:2] = T[1:height * 2:2, 0:width * 2:2]
    print ("image interpolation ", T.shape)

    I =  np.clip(I,0, 1)
    # **gamma correction**
    gamma = 0.6060
    I = I**gamma
    I = np.round(I*255)
    ##print ("**** ", I[27,27])
    I= np.uint8(I)

    RGB = cv2.demosaicing(I, cv2.COLOR_BayerBG2RGB_VNG)
    img = cv2.resize(RGB, (width,height),interpolation=cv2.INTER_CUBIC)
    return img

def imadjust(iChannel):
    iChannel = np.uint8(iChannel*255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imCha = clahe.apply(iChannel)

    imCha = np.float32(imCha)/255
    return imCha