"""

"""

__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"

import tensorflow as tf
import tensorflow.keras as tfk
import platform
import argparse

import time

from utls import (h5_reader,
                   mse, ssim_psnr,
                   h5_writer)
from models.cdent import CDENT
from utilities.data_manager import *
from dataset_manager import DataLoader



parser = argparse.ArgumentParser(description="CDNet arguments")

data_dir = '/opt/dataset' if platform.system()=="Linux" else '../../dataset'

parser.add_argument('--img_width', type=int,default=192,help="Image width")
parser.add_argument('--img_height', type=int,default=192,help="Image height")
parser.add_argument('--dataset_name',type=str, default= 'OMSIV', help="Dataset used by nir_cleaner choice [omsiv or ssmihd]")
parser.add_argument("--model_name",type=str, default='CDNet',help="Choise one of [CDNet, ENDENet]")
parser.add_argument('--num_channels',type=int, default= 3,help="The number of channels in the images to process")
parser.add_argument('--batch_size',type=int, default= 8,help="The size of the mini-batch")

parser.add_argument('--num_epochs',type=int, default= 100,help="The number of iterations during the training")
parser.add_argument('--margin', type=float, default=1.0,help="The margin value for the loss function")
parser.add_argument('--lr', type=float, default=1e-4,help="The learning rate for the SGD optimization")
parser.add_argument('--weight_decay', type=float, default=0.0002, help="Set the weight decay")
parser.add_argument('--use_base_dir', type=bool, default=False, help="True when you are going to put the base directory of OMSIV dataset")
parser.add_argument('--dataset_dir', type=str, default=data_dir)

parser.add_argument('--train_list', type=str, default='train_list.txt', help="File which contian the training data")
parser.add_argument('--test_list', type=str, default='test_list.txt', help="File which contain the testing data")
parser.add_argument('--gpu_id', type=str, default='0',help="The default GPU id to use")
parser.add_argument('--model_state', type=str, default='train',help="training or testing [train, test, None]")
parser.add_argument('--ckpt_dir', type=str, default='checkpoints',help="training or testing [True or False]")
parser.add_argument('--use_nir',type=bool, default=False,help="True for using the NIR channel")

arg = parser.parse_args()


def train():
    if arg.model_name.lower() =="cdnet" or arg.model_name.lower()=="endenet":
        # ***************data preparation *********************
        if arg.model_state.lower()=='train':


            # dataset preparation for training
            running_mode = 'train'
            data4training = DataLoader(
                data_name=arg.dataset_name,arg=arg)
            # define model and callbacks
            model_dir = arg.model_name.lower()+'2'+arg.dataset_name
            ckpnt_dir = os.path.join(arg.ckpt_dir,model_dir)
            ckpnt_path =os.path.join(ckpnt_dir,'saved_weights.h5')
            os.makedirs(ckpnt_dir,exist_ok=True)
            log_dir = os.path.join('logs',model_dir)
            res_dir = os.path.join('results', model_dir)
            os.makedirs(res_dir, exist_ok=True)

            my_callbacks = [
                tfk.callbacks.ModelCheckpoint(
                    ckpnt_path, monitor='train_loss',# os.path.join(ckpnt,saved_weights.h5)
                    save_weights_only=True, mode='auto',save_freq='epoch'),
                tfk.callbacks.TensorBoard(
                    log_dir,histogram_freq=0,write_graph=True,
                    profile_batch=2, write_images=True)
            ]
            my_model = CDENT()

            loss_mse = tfk.losses.mean_squared_error
            accuracy = tfk.metrics.MeanAbsolutePercentageError()
            optimizer =tfk.optimizers.Adam(learning_rate=arg.lr,
                                           beta_1=0.5)
            # compile model
            # my_model.compile(optimizer=optimizer, loss=loss_mse)
            # my_model.fit(data4training, epochs=arg.num_epochs,callbacks=my_callbacks)

            for epoch in range(arg.num_epochs):
                total_loss = tf.Variable(0.)
                for step,(x,y) in enumerate(data4training):

                    with tf.GradientTape() as tape:
                        p = my_model(x)
                        loss = loss_mse(y_true=y,y_pred=p)
                        loss = tf.math.reduce_sum(loss)
                    total_loss = tf.add(total_loss,loss)
                    accuracy.update_state(y_true=y,y_pred=p)
                    gradients = tape.gradient(loss,my_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients,my_model.trainable_variables))

                    if step % 10 == 0:
                        print("Epoch:", epoch, "Step:", step, "Loss: %.4f" % loss.numpy(),
                              "Accuracy: %.4f" % accuracy.result(), time.ctime())

                tfk.Model.save_weights(my_model, ckpnt_path, save_format='h5')
                print('Model saved in:',ckpnt_path)

                # visualize result
                mean_loss = total_loss / 50
                tmp_x = image_normalization(np.squeeze(x[2,:,:,:3]))
                tmp_y = image_normalization(np.squeeze(y[2,...]))
                tmp_p = p[2,...]
                tmp_p = image_normalization(tmp_p.numpy())
                vis_imgs = np.uint8(np.concatenate((tmp_x,tmp_y,tmp_p),axis=1))
                img_test = 'Epoch: {0}  Loss: {1}'.format(epoch, mean_loss.numpy())
                BLACK = (0, 0, 255)
                font = cv.FONT_HERSHEY_SIMPLEX
                font_size = 1.1
                font_color = BLACK
                font_thickness = 2
                x, y = 30, 30
                vis_imgs = cv.putText(vis_imgs, img_test, (x, y), font, font_size, font_color, font_thickness,
                                      cv.LINE_AA)
                cv.imwrite(os.path.join(res_dir, 'results.png'), vis_imgs)

                print("<<< End epoch loss: ",mean_loss.numpy()," >>>")

            my_model.summary()
            print('Training finished on: ', arg.dataset_name)



if __name__=="__main__":

    train()