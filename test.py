"""

"""
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import sys
# from tensorlayer.layers import *

import matplotlib.pyplot as plt
from tqdm import tqdm

def test_model(FLAGS):

    start_hyperparameters()
    try:
        img = get_imgs_fn(file)
    except IOError:
        print('cannot open %s' % (file))
    else:

        if is_tf:
            # for tensorflow based model

            save_dir = "%s/%s_%s" % (config.model.result_path, config.model.name, tl.global_flag['mode'])
            checkpoint_dir = "%s/%s" % (config.model.checkpoint_path, config.model.name)
            tl.files.exists_or_mkdir(save_dir)
            tl.files.exists_or_mkdir(checkpoint_dir)
            input_imagex0 = normalization_data_01(img)

            size = input_imagex0.shape
            print("Input size: ", input_imagex0.shape)
            input_tensor = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
            pred_t0, pred_t2, predt4 = srnet.net(input_tensor, is_train=False, reuse=False)

            ###========================== RESTORE G =============================###
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            tl.layers.initialize_global_variables(sess)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/params_train.npz', network=net_g)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + config.model.name + '/params_{}.npz'.format(
                tl.global_flag['mode']), network=predt4)

            ###======================= TEST =============================###

            start_time = time.time()
            out = sess.run(predt4.outputs, {input_tensor: [input_imagex0]})
            print("took: %4.4fs" % (time.time() - start_time))


        else:
            # for tensorflow and tensorlayer based model

            save_dir = "s%/s%_s%" % (config.model.result_path, config.model.name, tl.global_flag['mode'])
            checkpoint_dir = "s%/s%" % (config.model.checkpoint_path, config.model.name)
            input_image = normalize_imgs_fn(img)

            size = input_image.shape
            print('Input size: %s,%s,%s' % (size[0], size[1], size[2]))
            input_tensor = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
            pred_t0, pred_t2, predt4 = srnet.net(input_tensor, is_train=False, reuse=False)

            ###========================== RESTORE G =============================###
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            tl.layers.initialize_global_variables(sess)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/params_train.npz', network=net_g)
            tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + config.model.name + '/params_{}.npz'.format(
                tl.global_flag['mode']), network=predt4)

            ###======================= TEST =============================###

            start_time = time.time()
            out = sess.run(predt4.outputs, {input_tensor: [input_image]})
            print("took: %4.4fs" % (time.time() - start_time))

        tl.files.exists_or_mkdir(save_dir)
        tl.vis.save_image(truncate_imgs_fn(out[0, :, :, :]), save_dir + '/test_out.png')
        tl.vis.save_image(input_image, save_dir + '/test_input.png')