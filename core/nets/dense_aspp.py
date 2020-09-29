
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import BilinearUpSampling, bn_act_conv_block
from ..encoder import scope_table, build_encoder


def DenseASPP(input_shape,
              n_class,
              encoder_name,
              encoder_weights=None,
              weight_decay=1e-4,
              kernel_initializer="he_normal",
              bn_epsilon=1e-3,
              bn_momentum=0.99):
    """ implementation of Dense-ASPP for semantic segmentation.
        ref: Yang M, Yu K, Zhang C, et al. Denseaspp for semantic segmentation in street scenes[C].
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 3684-3692.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    init_features = encoder.get_layer(scope_table[encoder_name]["pool3"]).output

    ### First block, rate = 3
    d_3_features = bn_act_conv_block(init_features, n_filters=256, kernel_size=(1, 1),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    d_3 = bn_act_conv_block(d_3_features, n_filters=64, rate=3, kernel_size=(3, 3),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,