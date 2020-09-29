
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import BilinearUpSampling, bn_act_convtranspose, bn_act_conv_block
from ..encoder import scope_table, build_encoder


def residual_conv_unit(inputs,
                       n_filters=256,
                       kernel_size=3,
                       weight_decay=1e-4,
                       kernel_initializer="he_normal",
                       bn_epsilon=1e-3,
                       bn_momentum=0.99):
    """ residual convolutional unit.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of filters, default 256.
    :param kernel_size: int, default 3.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = Activation("relu")(inputs)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Add()([x, inputs])

    return x


def chained_residual_pooling(inputs,
                             pool_size=(5, 5),
                             n_filters=256,
                             weight_decay=1e-4,
                             kernel_initializer="he_normal",
                             bn_epsilon=1e-3,
                             bn_momentum=0.99):
    """ chained residual pooling.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param pool_size: tuple, default (5, 5).
    :param n_filters: int, number of filters, default 256.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x_relu = Activation("relu")(inputs)

    x = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x_relu)
    x = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x_sum1 = Add()([x_relu, x])

    x = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x)
    x = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x_sum2 = Add()([x, x_sum1])

    return x_sum2


def multi_resolution_fusion(high_inputs=None,
                            low_inputs=None,
                            n_filters=256,
                            weight_decay=1e-4,
                            kernel_initializer="he_normal",
                            bn_epsilon=1e-3,
                            bn_momentum=0.99):
    """ fuse multi resolution features.
    :param high_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with high spatial resolutions.
    :param low_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with low spatial resolutions.
    :param n_filters: int, number of filters, default 256.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    if high_inputs is None:
        fuse = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                      kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(low_inputs)
        fuse = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(fuse)
    else:
        conv_low = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                          kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(low_inputs)
        conv_low = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(conv_low)
        conv_high = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                           kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_inputs)
        conv_high = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(conv_high)
        conv_low = BilinearUpSampling(target_size=(int(conv_high.shape[1]), int(conv_high.shape[2])))(conv_low)
        fuse = Add()([conv_high, conv_low])