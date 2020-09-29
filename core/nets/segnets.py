
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.models import Model
from keras.regularizers import l2

from ..encoder import scope_table, build_encoder


def conv_bn_pool(inputs,
                 ConvBN_count=2,
                 n_filters=64,
                 pooling=True,
                 weight_decay=1e-4,
                 kernel_initializer="he_normal",
                 bn_epsilon=1e-3,
                 bn_momentum=0.99):
    """ Conv + BN + Pooling
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param ConvBN_count: int, number of Conv+BN, default 2.
    :param n_filters: int, number of filters, default 64.
    :param pooling: bool, whether to apply pooling, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = inputs
    for i in range(ConvBN_count):
        x = Conv2D(n_filters, (3, 3), padding='same', activation='relu',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)

    if pooling:
        x = MaxPooling2D()(x)
    return x


def SegNet(input_shape,
           n_class,
           encoder_name,
           encoder_weights=None,
           weight_decay=1e-4,
           kernel_initializer="he_normal",
           bn_epsilon=1e-3,
           bn_momentum=0.99):
    """ implementation of SegNet for semantic segmentation.
        ref: Badrinarayanan V, Kendall A, Cipolla R. SegNet: A Deep Convolutional Encoder-Decoder Architecture
        for Image Segmentation[J]. arXiv preprint arXiv:1511.00561, 2015.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.