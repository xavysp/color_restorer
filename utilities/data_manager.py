
"""

"""
import numpy as np
import sys, os
import cv2 as cv
from PIL import Image

from utls import *


def cv_imshow(title,img):
    print(img.shape)
    cv.imshow(title,img)
    cv.moveWindow(title,20,20)
    cv.waitKey(0)
    cv.destroyAllWindows()

def dataset_spliter(list_file, list_path, base_dir=False):
    """
    Given a list of the dataset path, dataset_spliter()
    splits the data for the training and validation.
    The validation part is extracted randomly 10% of the whole of data
    :param list_file:
    :param list_path:
    :param base_dir: it is understood that the directory base path is
    list_path and if so base_dir have to be no True
    :return:
    """
    files_name = read_files_list(os.path.join(list_path,list_file))

    if not base_dir:
        files_name = [c.split(' ') for c in files_name]
    else:
        files_name = [c.split(' ') for c in files_name]
        files_name = [(os.path.join(list_path, c[0]),
                       os.path.join(list_path, c[1])) for c in files_name]
    n = len(files_name)
    all_index = np.random.permutation(n)
    val_percentaje= int(n*0.1)
    train_index = all_index[:-val_percentaje]
    val_index = all_index[-(val_percentaje+1):-1]
    if n!=(len(val_index)+len(train_index)):
        print("Inconsistence in the size of train+validation and total data")
        sys.exit()
    cache_info={
        "files_path":files_name,
        "n_files": n,
        "files_indeces": all_index,
        "train_indices": train_index,
        "validation_indices": val_index
    }
    return cache_info

def data_augmentation(x1,y1,x2,y2, batch_size=None,img_size=None, nir=False):
    """
    Four images have to be entered, and the x means that
    the images are corrupted or the data which is going to
    pass through the red and the y are their respective ground truth.
    the size of the image are quadratic the width and height are the
    same and are setted by 128x218. For example,
    if the given image is 320x580, 16 sub-images are extracted from the
    following size x[35:320-35,40:580-40]
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    x2 = np.fliplr(x2)
    y2 = np.fliplr(y2)
    img_width = x1.shape[1]
    img_height = x1.shape[0]
    x_idx = np.arange(35, img_height - (35 + img_size), (img_height - (70 + img_size))/(batch_size//2))
    x_idx = np.array(x_idx, dtype=np.int32)
    y_idx = np.arange(40, img_width - (40 + img_size), (img_width - (80 + img_size))/(batch_size//2))
    y_idx = np.array(y_idx, dtype=np.int32)

    if len(x_idx)==len(y_idx)==(batch_size)//2:
        pass
    else:
        print("Error... generating indeces")
        x_idx=[35, 37, 42, 55, 67, 77, 81, 91, 97, 105, 110, 119, 122, 130, 132, 157]
        y_idx=[40, 65, 72, 99, 105, 120, 150, 190, 220, 236, 249, 251, 280, 308, 345, 388]
        if not len(x_idx)==len(y_idx)==(batch_size)//2:
            print("Error generating indeces for data augmentatio")
            sys.exit()
    np.random.shuffle(x_idx)
    np.random.shuffle(y_idx)
    x=np.zeros((batch_size,img_size,img_size,3))if not nir else np.zeros((batch_size,img_size,img_size,4))
    y = np.zeros((batch_size,img_size,img_size,3))
    j=0
    for i in range(0,batch_size,2):
        x[i,:,:,:] = x1[x_idx[j]:x_idx[j]+img_size,y_idx[j]:y_idx[j]+img_size,:]
        y[i, :, :, :] = y1[x_idx[j]:x_idx[j]+img_size, y_idx[j]:y_idx[j]+img_size, :]

        x[i+1, :, :, :] = x2[x_idx[j]:x_idx[j]+img_size, y_idx[j]:y_idx[j]+img_size, :]
        y[i+1, :, :, :] = y2[x_idx[j]:x_idx[j]+img_size, y_idx[j]:y_idx[j]+img_size, :]
        j+=1
    # for i in range(batch_size):
    #     a=np.squeeze(x[i,:,:,:])
    #     b= np.squeeze(y[i,:,:,:])
    #     a=img_post_processing(a)
    #     b = img_post_processing(b)
    #     tmp_img = np.concatenate((a,b))
    #     cv_imshow(str(i),tmp_img)
    return x,y

def open_images(file_list):
    if len(file_list)>2 and not len(file_list)==3:
        imgs=[]
        file_names = []
        for i in range(len(file_list)):
            tmp = Image.open(file_list[i])
            imgs.append(tmp)
            file_names.append(file_list[i])

    elif len(file_list)>2 and len(file_list)==3:

        imgs = Image.open(file_list[2])
        file_names = file_list
    else:
        imgs = Image.open(file_list[1])
        file_names= file_list

    return imgs, file_names


# _____________ End batch management

# _____________ Save result _________
def save_result(config, data):
    """
    :param config:  model configs
    :param data: [input_data, label_data, predi_data]
    :return: just print done
    """
    main_dir = '/home/xsoria/matlabprojects/edges/results'

    if  config.which_dataset =="SSMIHD" and config.use_nir:
        main_dir = os.path.join(main_dir,config.dataset_name+'_RGBN')
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

    elif config.which_dataset =="SSMIHD" and not config.use_nir:
        main_dir = os.path.join(main_dir, config.dataset_name)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

    elif config.which_dataset=="HED-BSDS":
        main_dir = os.path.join(main_dir, config.result_folder)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
    else:
        print_error("something is wrong with the dataset implementation, check it ;)")
        sys.exit()

    if config.model_state=="train":
        file_path= os.path.join(main_dir,'valid_res.h5')
    elif config.model_state=="test":
        file_path = os.path.join(main_dir, 'test_res.h5')
    else:
        print_error("there is not any model state")
        sys.exit()

    in_data = data[0]
    label = np.array(data[1])
    predi = data[2]
    h5_writer(file_path, in_data,label,predi)