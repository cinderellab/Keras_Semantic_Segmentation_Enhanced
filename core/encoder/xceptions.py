
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.models import Model
from keras.regularizers import l2


def separable_residual_block(inputs,
                           n_filters_list=[256, 256, 256],
                           block_id="entry_block2",
                           skip_type="sum",
                           stride=1,
                           rate=1,
                           weight_decay=1e-4,
                           kernel_initializer="he_normal",
                           bn_epsilon=1e-3,
                           bn_momentum=0.99):
    """ separable residual block
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters_list: list of int, numbers of filters in the separable convolutions, default [256, 256, 256].
    :param block_id: string, default "entry_block2".
    :param skip_type: string, one of {"sum", "conv", "none"}, default "sum".
    :param stride: int, default 1.
    :param rate: int, default 1.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = Activation("relu", name=block_id+"sepconv1_act")(inputs)
    x = SeparableConv2D(n_filters_list[0], (3, 3), padding='same', use_bias=False,
                        name=block_id+'_sepconv1', dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+'_sepconv1_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)

    x = Activation('relu', name=block_id+'_sepconv2_act')(x)
    x = SeparableConv2D(n_filters_list[1], (3, 3), padding='same', use_bias=False,
                        name=block_id+'_sepconv2', dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+'_sepconv2_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)

    x = Activation("relu", name=block_id+"_sepconv3_act")(x)
    x = SeparableConv2D(n_filters_list[2], (3, 3), padding="same", use_bias=False,
                        strides=stride, name=block_id+"_sepconv3", dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+"_sepconv3_bn", epsilon=bn_epsilon, momentum=bn_momentum)(x)

    if skip_type=="sum":
        x = Add()([inputs, x])