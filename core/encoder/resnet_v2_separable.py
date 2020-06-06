from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.regularizers import l2


def residual_block(inputs,
                   base_depth,
                   depth,
                   kernel_size,
                   stride=1,
                   rate=1,
                   block_name="block1",
                   unit_name="