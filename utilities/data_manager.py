
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

def data_parser(args, files_path=None, indx1=None, indx2=None,batch_size=None):

    if args.model_state=='Train' or args.model_state=='train':

        if args.dataset_name=="SSMIHD" or args.dataset_name=="ssmihd":
            # reading data and normalization in a range of 0-1
            x1 =h5_reader(files_path[indx1][0])
            x1=normalization_data_01(x1[0])
            y1 = h5_reader(files_path[indx1][1])
            y1 = normalization_data_01(y1[0])
            x2 = h5_reader(files_path[indx2][0])
            x2 = normalization_data_01(x2[0])
            y2 = h5_reader(files_path[indx2][1])
            y2 = normalization_data_01(y2[0])
            if args.use_nir:
                pass
            else:
                x1=x1[:,:,:3]
                x2 = x2[:,:,:3]
            x, y = data_augmentation(x1,y1,x2,y2, batch_size=args.batch_size,
                                     img_size=args.image_size, nir=args.use_nir)

            return x,y

        elif args.dataset_name=="omsiv" or args.dataset_name=="OMSIV":
            # reading data and normalization in a range of 0-1
            x1 =h5_reader(files_path[indx1][0])
            x1=normalization_data_01(x1[0])
            y1 = h5_reader(files_path[indx1][1])
            y1 = normalization_data_01(y1[0])
            x2 = h5_reader(files_path[indx2][0])
            x2 = normalization_data_01(x2[0])
            y2 = h5_reader(files_path[indx2][1])
            y2 = normalization_data_01(y2[0])
            if args.use_nir:
                pass
            else:
                x1=x1[:,:,:3]
                x2 = x2[:,:,:3]
            x, y = data_augmentation(x1,y1,x2,y2, batch_size=args.batch_size,
                                     img_size=args.image_size, nir=args.use_nir)

            return x,y
        else:
            print("There is not other dataset loader implemented yet")
            return

# ************** for testing **********************
    elif args.model_state=='test':
        pass

        # if args.which_dataset == "SSMIHD":
        #     if args.test_augmented:
        #         valid_files_name = os.path.join('valid', args.val_list)
        #     else:
        #         valid_files_name = os.path.join('valid', args.test_list)
        #     if args.use_nir:
        #         valid_list_path = os.path.join(args.dataset_dir+args.dataset_name,
        #                                        valid_files_name)
        #
        #         valid_list = read_files_list(valid_list_path)
        #         valid_list = split_pair_names(valid_list,
        #                                       dataset_name=args.which_dataset, use_nir=True)
        #         n_valid = int(len(valid_list))
        #         print_info("Validation set-up from {}, size: {}".format(valid_list_path, n_valid))
        #         if args.test_augmented:
        #             valid_ids = np.arange(int(n_valid // 2), n_valid)
        #         else:
        #             valid_ids = np.arange(n_valid)
        #
        #         print_info("Test set-up from {}, size: {}".format(valid_list_path, len(valid_ids)))
        #
        #         # np.random.shuffle(valid_ids)
        #         cache_out = [valid_list, valid_ids]
        #
        #     else:
        #         valid_list_path = os.path.join(args.dataset_dir + args.dataset_name,
        #                                    valid_files_name)
        #         valid_list = read_files_list(valid_list_path)
        #
        #         valid_list = split_pair_names(valid_list)
        #         n_valid = int(len(valid_list))
        #         print_info("testing set-up from {}, size: {}".format(valid_list_path, n_valid))
        #
        #         if args.test_augmented:
        #             valid_ids = np.arange(int(n_valid // 2), n_valid)
        #         else:
        #             valid_ids = np.arange(n_valid)
        #         # np.random.shuffle(valid_ids)
        #         cache_out = [valid_list, valid_ids]
        #     return cache_out
        #
        # elif args.which_dataset == "HED-BSDS":
        #     test_files_name = args.test_list
        #     test_list_path = os.path.join(args.dataset_dir + args.which_dataset,
        #                                    test_files_name)
        #     test_list = read_files_list(test_list_path)
        #
        #     test_samples = split_pair_names(test_list, args.dataset_dir + args.which_dataset)
        #     n_test = len(test_samples)
        #     print_info(" Enterely testing set-up from {}, size: {}".format(test_list_path, n_test))
        #
        #     test_ids = np.arange(n_test)
        #     # np.random.shuffle(test_ids)
        #
        #     print_info("testing set-up from {}, size: {}".format(test_list_path, len(test_ids)))
        #     cache_out = [test_samples, test_ids]
        #     return cache_out
        #
        # else:
        #     # for NYUD
        #     test_files_name = args.test_list
        #     test_list_path = os.path.join(args.dataset_dir + args.which_dataset,
        #                                   test_files_name)
        #     test_list = read_files_list(test_list_path)
        #
        #     test_samples = split_pair_names(test_list, args.dataset_dir + args.which_dataset,
        #                                     dataset_name=args.which_dataset)
        #     n_test = len(test_samples)
        #     print_info(" Enterely testing set-up from {}, size: {}".format(test_list_path, n_test))
        #
        #     test_ids = np.arange(n_test)
        #     np.random.shuffle(test_ids)
        #
        #     print_info("testing set-up from {}, size: {}".format(test_list_path, len(test_ids)))
        #     cache_out = [test_samples, test_ids]
        #     return cache_out
    else:
        print_error("The model state is just train and test")
        sys.exit()

def test_data_loader(args):
    dataset_dir = os.path.join(FLAGS.dataset_dir,
                               os.path.join(FLAGS.dataset_name, FLAGS.task))
    dataset_dir = os.path.join(dataset_dir, 'X') if FLAGS.use_all_data else \
        os.path.join(dataset_dir, os.path.join(FLAGS.model_state, 'X'))

    data_list = os.listdir(dataset_dir)
    data_list.sort()
    return data_list
# ___________batch management ___________
def get_batch(args,file_list=None, current_indx = None, indcs=None, use_batch=True):
    dataset_dir = os.path.join(args.dataset_dir, os.path.join(args.dataset_name,
                                                              args.task))
    dataset_dir = os.path.join(dataset_dir, os.path.join(args.model_state, 'X'))

    if use_batch:
        file_names =[]
        images=[]
        edgemaps=[]
        if args.use_nir:
            for idx, b in enumerate(args.batch_size):
                x_nir = Image.open(file_list[b][0])
                x_rgb = Image.open(file_list[b][1])
                y = Image.open(file_list[b][2])

                x_nir = x_nir.resize((args.image_width, args.image_height))
                x_rgb = x_rgb.resize((args.image_width, args.image_height))
                y = y.resize((args.image_width, args.image_height))

                x_nir = x_nir.convert("L")
                # pay attention here
                x_nir = np.array(x_nir, dtype=np.float32)
                x_nir = np.expand_dims(x_nir,axis=2)
                x_rgb = np.array(x_rgb, dtype=np.float32)

                x_rgb = x_rgb[:, :, args.channel_swap]
                x = np.concatenate((x_rgb,x_nir),axis=2)
                x -= args.mean_pixel_value

                y = np.array(y.convert('L'), dtype=np.float32)
                if args.target_regression:
                    bin_y = y/255.0
                else:
                    bin_y = np.zeros_like(y)
                    bin_y[np.where(y)]=1

                bin_y = bin_y if bin_y.ndim ==2 else bin_y[:,:,0]
                bin_y = np.expand_dims(bin_y,axis=2)

                images.append(x)
                edgemaps.append(bin_y)
                file_names.append(file_list[b])

        else:
            if args.dataset_name=='omsiv' or args.dataset_name=='OMSIV':

                data = h5_reader(os.path.join(dataset_dir, file_list[0]))
                print(data.shape)
                im_width = data.shape[1]
                im_height = data.shape[0]
                imgs= np.zeros((FLAGS.batch_size,im_height,im_width,
                                3)) if args.use_nir else np.zeros((FLAGS.batch_size,
                                                                   im_height,im_width,3))
                imgs_name=[]
                n = current_indx+args.batch_size if current_indx+args.batch_size<len(file_list) else len(file_list)
                j = 0
                for i in range(current_indx,n):
                    tmp_img = h5_reader(os.path.join(dataset_dir, file_list[i]))
                    imgs[j,:,:,:]= tmp_img[:,:,:] if args.use_nir else tmp_img[:,:,0:3]
                    j+=1
                    imgs_name.append(file_list[i])
                return imgs. imgs_name

            else:

                for idx, b in enumerate(args.batch_size):
                    x = Image.open(file_list[b][0])
                    y = Image.open(file_list[b][1])

                    x = x.resize((args.image_width, args.image_height))
                    y = y.resize((args.image_width, args.image_height))
                    # pay attention here
                    x = np.array(x, dtype=np.float32)
                    x = x[:, :, args.channel_swap]
                    x -= args.mean_pixel_value[0:3]

                    y = np.array(y.convert('L'), dtype=np.float32)
                    if args.target_regression:
                        bin_y = y/255.0
                    else:
                        bin_y = np.zeros_like(y)
                        bin_y[np.where(y)]=1

                    bin_y = bin_y if bin_y.ndim ==2 else bin_y[:,:,0]
                    bin_y = np.expand_dims(bin_y,axis=2)

                    images.append(x)
                    edgemaps.append(bin_y)
                    file_names.append(file_list[b])

                return images, edgemaps, file_names

    else:
        if args.which_dataset=='NYUD' and args.model_state=='test':

            x = Image.open(file_list)
            if args.image_height%2==0:
                pass
            else:
                x = x.resize((args.image_width, args.image_height+7))
            # pay attention here
            x = np.array(x, dtype=np.float32)
            x = x[:, :, args.channel_swap]
            x -= args.mean_pixel_value[:-1]

            images = x
            file_names = file_list
            edgemaps = None

        elif args.which_dataset=='SSMIHD' and (args.model_state=='test' and args.use_nir):
            x_nir = Image.open(file_list[0])
            x_rgb = Image.open(file_list[1])
            # y = Image.open(file_list[2])
            if not args.image_width % 6 == 0 and args.image_width > 1000:
                x_nir = x_nir.resize((args.image_width, args.image_height)) #--
                x_rgb = x_rgb.resize((args.image_width, args.image_height)) # --
            else:
                x_nir = x_nir.resize((args.image_width, args.image_height))
                x_rgb = x_rgb.resize((args.image_width, args.image_height))

            # y = y.resize((arg.image_width, arg.image_height))

            x_nir = x_nir.convert("L")
            # pay attention here
            x_nir = np.array(x_nir, dtype=np.float32)
            x_nir = np.expand_dims(x_nir, axis=2)
            x_rgb = np.array(x_rgb, dtype=np.float32)

            x_rgb = x_rgb[:, :, args.channel_swap]
            x = np.concatenate((x_rgb, x_nir), axis=2)
            x -= args.mean_pixel_value
            # y = np.array(y.convert('L'), dtype=np.float32)
            # if arg.target_regression:
            #     bin_y = y / 255.0
            # else:
            #     bin_y = np.zeros_like(y)
            #     bin_y[np.where(y)] = 1
            #
            # bin_y = bin_y if bin_y.ndim == 2 else bin_y[:, :, 0]
            # bin_y = np.expand_dims(bin_y, axis=2)

            images =x
            edgemaps  = None
            file_names = file_list[2]

        else:
            if args.which_dataset == 'SSMIHD':
                x = Image.open(file_list[1])
                y = Image.open(file_list[2])
            else:
                x = Image.open(file_list[0])
                y = Image.open(file_list[1])

            if not args.image_width % 16 == 0 and args.image_width > 1000:
                x = x.resize((args.image_width, args.image_height)) # -8
                y = y.resize((args.image_width, args.image_height)) # -8 width
            else:
                x = x.resize((args.image_width, args.image_height))
                y = y.resize((args.image_width, args.image_height))
            # pay attention here
            x = np.array(x, dtype=np.float32)
            x = x[:, :, args.channel_swap]
            x -= args.mean_pixel_value[:-1]

            y = np.array(y.convert('L'), dtype=np.float32)
            if args.target_regression:
                bin_y = y / 255.0
            else:
                bin_y = np.zeros_like(y)
                bin_y[np.where(y)] = 1

            bin_y = bin_y if bin_y.ndim == 2 else bin_y[:, :, 0]
            bin_y = np.expand_dims(bin_y, axis=2)

            images = x
            edgemaps = bin_y
            file_names = file_list[1]

        return images, edgemaps, file_names


def get_training_batch(arg, list_ids):
    if arg.dataset_name=="SSMIHD":
        train_ids = list_ids[2]
        file_list = list_ids[0]
        batch_ids = np.random.choice(train_ids, arg.batch_size_train)

    else:
        train_ids = list_ids[1]
        file_list= list_ids[0]
        batch_ids = np.random.choice(train_ids,arg.batch_size_train)
    return get_batch(arg, file_list, batch_ids)

def get_validation_batch(arg, list_ids):
    if arg.dataset_name =="SSMIHD":
        valid_ids = list_ids[-1]
        file_list = list_ids[1]
        batch_ids = np.random.choice(valid_ids, arg.batch_size_val)
    else:
        valid_ids = list_ids[-1]
        file_list= list_ids[0]
        batch_ids = np.random.choice(valid_ids,arg.batch_size_val)
    return get_batch(arg,file_list,batch_ids)

def get_testing_batch(args,file_list=None, current_indx = None):

    dataset_dir = os.path.join(args.dataset_dir, os.path.join(args.dataset_name,
                                                              args.task))
    dataset_dir = dataset_dir = os.path.join(dataset_dir,'X') if args.use_all_data else \
        os.path.join(dataset_dir, os.path.join(FLAGS.model_state,'X'))

    if args.batch_size>1:

        if (args.dataset_name == 'omsiv' or args.dataset_name == 'OMSIV')or(args.dataset_name == 'ssmihd' or args.dataset_name == 'SSMIHD') :

            data = h5_reader(os.path.join(dataset_dir, file_list[0]))
            data = data[0]
            print(data.shape)
            im_width = data.shape[1]
            im_height = data.shape[0]
            imgs = np.zeros((FLAGS.batch_size, im_height, im_width,
                             3)) if args.use_nir else np.zeros((FLAGS.batch_size,
                                                                im_height, im_width, 3))
            imgs_name = []
            n = current_indx + args.batch_size if current_indx + args.batch_size < len(file_list) else len(file_list)
            j = 0
            for i in range(current_indx, n):
                tmp_img = h5_reader(os.path.join(dataset_dir, file_list[i]))
                tmp_img=normalization_data_01(tmp_img[0])
                imgs[j, :, :, :] = tmp_img[:, :, :] if args.use_nir else tmp_img[:, :, 0:3]
                j += 1
                # if i< len(file_list):
                imgs_name.append(file_list[i])
            return imgs, np.array(imgs_name)
    else:
        tmp_img = h5_reader(os.path.join(dataset_dir, file_list[current_indx]))
        tmp_img =normalization_data_01(tmp_img[0])
        img = tmp_img[:, :, :] if args.use_nir else tmp_img[:, :, 0:3]

        img_name = file_list[current_indx]
        return np.expand_dims(img, axis=0), img_name
    # if use_batch:
    #     test_ids = list_ids[1]
    #     file_list = list_ids[0]
    #     batch_ids = test_ids[i:i + arg.batch_size_test]
    #     return get_batch(arg,file_list,batch_ids)
    # else:
    #     if arg.which_dataset=='SSMIHD':
    #         return get_batch(arg, list_ids[0],list_ids[1], use_batch=False)
    #     else:
    #         return get_batch(arg, list_ids[0], list_ids[1], use_batch=False)


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