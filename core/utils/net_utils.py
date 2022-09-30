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

    def _resize_function(self, inputs):
        return tf.cast(tf.image.resize_bilinear(inputs, self.target_size, align_corners=True), dtype=tf.float32)

    def call(self, inputs):
        return self._resize_function(inputs=inputs)

    def get_config(self):
        config = {'target_size': self.target_size}
        base_config = super(BilinearUpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def separable_conv_bn(x,
                 