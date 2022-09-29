from keras.engine import Layer
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose, DepthwiseConv2D, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.regularizers import l2
import tensorflow as tf


class BilinearUpSampling(Layer):
    def __init__(self, target_size, **kwargs):
        super(BilinearUpSampling, self).__init__(**kwargs)
        se