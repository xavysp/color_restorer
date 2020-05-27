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