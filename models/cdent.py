"""

"""
import tensorflow as tf
import tensorflow.contrib.layers as lays


def net(ls, input, reuse=False, train=False):

    with tf.variable_scope("CDNet", reuse=reuse):
        net = tf.layers.conv2d(input, ls[1], [3, 3], strides=(1, 1), padding='SAME',
                           activation=tf.nn.relu, trainable=train) # n_batchx192x192x32
        print("conv1 ", net)

        net = lays.conv2d(net, ls[2], [3, 3], stride=(1, 1), padding="SAME",activation_fn=tf.nn.relu,
                          weights_initializer=lays.xavier_initializer(uniform=True),
                          trainable=train) #n_batchx192x192x64
        print("conv2 ", net)

        net1 = lays.conv2d_transpose(net, ls[3], [3, 3], stride=(1, 1), padding="SAME",
                                     activation_fn=tf.nn.relu, weights_initializer=lays.xavier_initializer(uniform=True),
                                     trainable=train) # n_batchx192x192x32
        print("deconv1 ", net1)

        net1 = lays.conv2d_transpose(net1, ls[4], [3, 3], stride=(1, 1), padding='SAME',
                                     activation_fn=tf.nn.relu, weights_initializer=lays.xavier_initializer(uniform=True),
                                     trainable=train)  # n_batchx192x192x64
        print("deconv2 ", net1)

        net1 = lays.conv2d_transpose(net1, ls[5], [3, 3], stride=(1, 1), padding='SAME',
                                     activation_fn=tf.nn.relu, weights_initializer=lays.xavier_initializer(uniform=True),
                                     trainable=train)  # n_batchx192x192x32
        print("deconv3 ", net1)  # ls[6]

        net1 = lays.conv2d_transpose(net1, ls[6], [1, 1], stride=(1, 1), padding='SAME',
                                     weights_initializer=lays.xavier_initializer(uniform=True),
                                     trainable=train)  # n_batchx192x192x3
        print("deconv4 ", net1)

    return net1, net