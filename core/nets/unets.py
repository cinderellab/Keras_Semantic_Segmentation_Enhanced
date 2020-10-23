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
    :param input_shape: tuple, i.e., 