

import tensorflow.keras as tfk


l2 = tfk.regularizers.l2
class ConvLayer(tfk.Model):
    def __init__(self, n_filters=32,k_size=(3,3),
                 strides=(1,1),weight_decay=1e4):
        super(ConvLayer, self).__init__()

        self.conv1 = tfk.layers.Conv2D(
            filters=n_filters, kernel_size=k_size,strides=strides,
            padding='same', kernel_initializer='glorot_uniform',
            activation='relu', kernel_regularizer=l2(weight_decay))

    def call(self, x, training=False):
        output = self.conv1(x)
        return output

class DconvLayer(tfk.Model):
    def __init__(self, n_filters=32,k_size=(3,3),
                 strides=(1,1),act='relu',weight_decay = 1e4):
        super(DconvLayer,self).__init__()

        self.dconv1 = tfk.layers.Conv2DTranspose(
            filters=n_filters, kernel_size=k_size,strides=strides,
            activation=act,padding='same',kernel_initializer='glorot_uniform',
            kernel_regularizer=l2(weight_decay)
        )

    def call(self, x, training=False):
        output = self.dconv1(x)
        return output

class CDENT(tfk.Model):
    def __init__(self):
        super(CDENT,self).__init__()

        self.conv1 = ConvLayer(32,(3,3),(1,1))
        self.conv2 = ConvLayer(64,(3,3),(1,1))
        self.dconv1 = DconvLayer(32,(3,3),(1,1))
        self.dconv2 = DconvLayer(64,(3,3),(1,1))
        self.dconv3 = DconvLayer(32,(3,3),(1,1))
        self.dconv4 = DconvLayer(3,(1,1),(1,1),act=None)

    def call(self,x,training=False):

        output = self.conv1(x)
        output = self.conv2(output)
        output = self.dconv1(output)
        output = self.dconv2(output)
        output = self.dconv3(output)
        output = self.dconv4(output)

        return output


