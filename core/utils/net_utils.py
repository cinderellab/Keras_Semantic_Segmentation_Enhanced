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
        self.target_size = target_size

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[-1])

  