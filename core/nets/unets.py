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

    conv1 = Conv2D(init_filters * 1, (3, 3), activatio