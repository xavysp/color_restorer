"""

"""
import tensorflow as tf


def net(ls, input, reuse=False, train=False):
    ######################
    # hl = [3,32,64,32,64,32,3] # CDED_L adam op
    with tf.variable_scope("ENDENet", reuse=reuse):

        net = tf.layers.conv2d(input, ls[1], [3, 3], strides=(2, 2), padding='SAME',
                               activation=tf.nn.relu, trainable=train)  # n_batchx192x192x32
        print("conv1 ", net)

        net = tf.layers.conv2d(net, ls[2], [3, 3], strides=(2, 2), padding='SAME',
                               activation=tf.nn.relu, trainable=train)  # n_batchx192x192x64
        print("conv2 ", net)

        net = tf.layers.conv2d_transpose(net, ls[3], [3, 3], strides=(1, 1), padding="SAME",
                                         activation=tf.nn.relu, trainable=train)  # n_batchx192x192x32
        print("deconv1 ", net)

        net1 = tf.layers.conv2d_transpose(net, ls[4], [3, 3], strides=(2, 2), padding="SAME",
                                          activation=tf.nn.relu, trainable=train)  # n_batchx192x192x64
        print("deconv2 ", net1)

        net1 = tf.layers.conv2d_transpose(net1, ls[5], [3, 3], strides=(2, 2), padding="SAME",
                                          activation=tf.nn.relu, trainable=train)  # n_batchx192x192x32
        print("deconv3 ", net1)  # ls[6]

        net1 = tf.layers.conv2d_transpose(net1, ls[6], [1, 1], strides=(1, 1), padding="SAME",
                                          activation=tf.nn.relu, trainable=train)  # n_batchx192x192x3
        print("deconv4 ", net1)

    return net1, net