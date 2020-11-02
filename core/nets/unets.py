from keras.engine import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import conv_bn_act_block, bn_act_convtranspose


def UNet(input_shape,
         n_class,
         weight_decay=1e-4,
         kernel_initializer="he_normal",
         bn_epsilon=1e-3,
         bn_momentum=0.99,
         init_filters=64,
         dropout=0.5):
    """ Implementation of U-Net for semantic segmentation.
        ref: Ronneberger O , Fischer P , Brox T . U-Net: Convolutional Networks for Biomedical Image Segmentation[J].
             arXiv preprint arXiv: 1505.04597, 2015.
    :param input_shape: tuple, i.e., (width, height, channel).
    :param n_class: int, number of classes, at least 2.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param init_filters: int, initial filters, default 64.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    conv1 = Dropout(dropout)(conv1)
    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool1)
    conv2 = Dropout(dropout)(conv2)
    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool2)
    conv3 = Dropout(dropout)(conv3)
    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool3)
    conv4 = Dropout(dropout)(conv4)
    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool4)
    conv5 = Dropout(dropout)(conv5)
    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv5)

    up1 = Concatenate()([Conv2DTranspose(init_filters * 8, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv5), conv4])
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(up1)
    conv6 = Dropout(dropout)(conv6)
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv6)

    up2 = Concatenate()([Conv2DTranspose(init_filters * 4, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv6), conv3])
    conv7 = 