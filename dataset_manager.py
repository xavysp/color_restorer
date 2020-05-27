import tensorflow as tf
import numpy as np
import h5py, os
import random


AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1024

img_shape =None

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self,data_name,arg=None):

        self.dim_w = arg.img_width # (arg.image_size,arg.image_size,3)
        self.dim_h = arg.img_height # (arg.image_size,arg.image_size,3)
        self.args = arg
        self.data_name =data_name
        self.bs = arg.batch_size
        self.is_training = True if arg.model_state.lower()=='train' else False
        self.shuffle=self.is_training
        self.data_list = self._build_index()
        self.on_epoch_end()
        # print(len(self.indices))
        if not self.is_training and arg.model_state=="test":
            i_width= arg.img_width if arg.img_width%4==0 else (arg.img_width//4+1)*4
            i_height= arg.img_height if arg.img_height%4==0 else (arg.img_height//4+1)*4
            self.input_shape = (None,i_height, i_width,4)
            self.imgs_shape=[]
            # OMSIV real size= 320,580,3

    def _build_index(self):

        base_dir = os.path.join(self.args.dataset_dir, self.data_name,self.args.model_state.lower())
        list_name= self.args.train_list if self.is_training else self.args.test_list
        file_path= os.path.join(base_dir,list_name)

        with open(file_path,'r') as f:
            file_list = f.readlines()
        file_list = [line.strip() for line in file_list] # to clean the '\n'
        file_list = [line.split(' ') for line in file_list] # separate paths

        input_path = [os.path.join(base_dir,line[0]) for line in file_list]
        gt_path = [os.path.join(base_dir,line[1]) for line in file_list]
        if not self.is_training:
            self.imgs_name = [os.path.basename(k) for k in input_path]
        sample_indeces= [input_path, gt_path]
        return sample_indeces

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)//self.bs


    def __getitem__(self, index):
        indices = self.indices[index*self.bs:(index+1)*self.bs]
        x_list,y_list = self.data_list
        tmp_x_path = [x_list[k] for k in indices]
        tmp_y_path = [y_list[k] for k in indices]

        x,y = self.__data_generation(tmp_x_path,tmp_y_path)
        return x,y

    def __data_generation(self,x_path,y_path):

        x = np.empty((self.bs,self.dim_h,self.dim_w,4),dtype="float32")
        y = np.empty((self.bs,self.dim_h,self.dim_w,3),dtype="float32")

        for i,tmp_data in enumerate(x_path):
            tmp_x_path = tmp_data
            tmp_y_path = y_path[i]
            tmp_x,tmp_y = self.transformer(tmp_x_path,tmp_y_path)
            x[i,]=tmp_x
            y[i,]=tmp_y
        return x,y

    def transformer(self, x_path, y_path):
        tmp_x = self.__read_h5(x_path)
        tmp_y = self.__read_h5(y_path)
        h,w,_ = tmp_x.shape
        if self.args.model_state == "train":
            i_h = random.randint(0,h-self.dim_h)
            i_w = random.randint(0,w-self.dim_w)
            tmp_x = tmp_x[i_h:i_h+self.dim_h,i_w:i_w+self.dim_w,]
            tmp_y = tmp_y[i_h:i_h+self.dim_h,i_w:i_w+self.dim_w,]

        return tmp_x, tmp_y

    def __read_h5(self,file_path):

        with h5py.File(file_path,'r') as h5f:
            # n_var = len(list(h5f.keys()))
            data = np.array(h5f.get('data'))
        return data


def load(image_file):

    input_img = tf.io.read_file(image_file[0])
    input_img = tf.image.decode_jpeg(input_img, channels=3)
    # opening target image in png
    gt_img = tf.io.read_file(image_file[1])
    gt_img = tf.image.decode_png(gt_img, channels=3)

    # input_img = tf.image.convert_image_dtype(input_img,tf.float32)
    # gt_img = tf.image.convert_image_dtype(gt_img,tf.float32)
    return input_img, gt_img


def resize(input_image, real_image):

    input_image = tf.image.resize(input_image, [img_shape[1], img_shape[0]])
    real_image = tf.image.resize(real_image, [img_shape[1], img_shape[0]])
    return input_image, real_image

def random_crop(input_img, real_img):

    stacked_img = tf.stack([input_img, real_img], axis=0)
    cropped_img = tf.image.random_crop(stacked_img, size=[2, img_shape[0], img_shape[1], 3])
    return cropped_img[0], cropped_img[1]


def pix2pix_norm(input_img, real_img):
    input_img = (input_img / 127.5) - 1
    real_img = (real_img / 127.5) - 1
    return input_img, real_img


# @tf.function()
def random_jitter(input_img, real_img):

    if np.random.random() > 0.5:
        # resize image
        input_img, real_img = resize(input_img,real_img)
    else:
        input_img, real_img = random_crop(input_img, real_img)

    return input_img, real_img

@tf.function
def load_train_img(input_paths,target_path):
    # input_paths, target_path = input_paths
    input_img, gt_img = load([input_paths, target_path])
    input_img, gt_img = random_jitter(input_img, gt_img)
    # normalize gt
    input_img = tf.image.convert_image_dtype(input_img,tf.float32)
    gt_img = tf.image.rgb_to_grayscale(gt_img)
    gt_img = tf.image.convert_image_dtype(gt_img,tf.float32)/255.

    return input_img, gt_img


def load_val_img(input_paths,target_path):
    # input_paths, target_path = input_paths
    input_img, gt_img = load([input_paths, target_path])
    input_img, gt_img = resize(input_img,gt_img)
    # normalize gt
    input_img = tf.image.convert_image_dtype(input_img,tf.float32)
    gt_img = tf.image.rgb_to_grayscale(gt_img)
    gt_img = tf.image.convert_image_dtype(gt_img, tf.float32) / 255.
    return input_img, gt_img

def load_test_img(image_paths):
    input_img = tf.io.read_file(image_paths)
    input_img = tf.image.decode_jpeg(input_img, channels=3)
    input_img = tf.image.resize(input_img, [img_shape[1], img_shape[0]])

    return input_img

def data_loader(data_list=None, arg=None, training=True):

    if arg.model_state.lower()=="train":
        image_shape = [arg.image_height, arg.image_width]
    else:
        image_shape = [arg.test_img_height, arg.test_img_width]
    global img_shape
    img_shape = image_shape
    if training:
        # for training
        img_tensor = tf.convert_to_tensor(data_list[:,0])
        lbl_tensor = tf.convert_to_tensor(data_list[:,1])
        train_data = tf.data.Dataset.from_tensor_slices((img_tensor,lbl_tensor))
        # train_data = tf.data.Dataset.from_tensor_slices((train_list[:,0],train_list[:,1]))
        train_data = train_data.shuffle(BUFFER_SIZE)
        train_data = train_data.map(load_train_img, num_parallel_calls=AUTOTUNE) # tf.data.experimental.
        train_data = train_data.repeat(arg.max_epochs)
        train_data = train_data.batch(arg.batch_size)
        train_data = train_data.prefetch(buffer_size=AUTOTUNE)
        return train_data
    else:
        if arg.model_state.lower()=="train":
            # for validation
            img_tensor = tf.convert_to_tensor(data_list[:, 0])
            lbl_tensor = tf.convert_to_tensor(data_list[:, 1])
            val_data = tf.data.Dataset.from_tensor_slices((img_tensor, lbl_tensor))
            # val_data = tf.data.Dataset.from_tensor_slices((val_list[:,0],val_list[:,1]))
            val_data = val_data.map(load_val_img)
            val_data = val_data.shuffle(24)
            val_data = val_data.repeat(arg.max_epochs)
            val_data = val_data.batch(arg.batch_size)
            val_data = val_data.prefetch(buffer_size=AUTOTUNE)
            # val_data = val_data.range(16)        # call models
            return val_data
        else:
            # for Testing
            img_tensor = tf.convert_to_tensor(data_list)
            test_data = tf.data.Dataset.from_tensor_slices(img_tensor)
            # val_data = tf.data.Dataset.from_tensor_slices((val_list[:,0],val_list[:,1]))
            test_data = test_data.map(load_test_img)
            test_data = test_data.repeat(1)
            test_data = test_data.batch(arg.test_bs)
            test_data = test_data.prefetch(buffer_size=AUTOTUNE)
            # val_data = val_data.range(8)  # call models
            return test_data