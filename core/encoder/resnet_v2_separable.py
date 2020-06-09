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
                   unit_name="unit1",
                   weight_decay=1e-4,
                   kernel_initializer="he_normal",
                   bn_epsilon=1e-3,
                   bn_momentum=0.99):
    """Implementation of a residual block, with 3 conv layers. Each convolutional layer is followed
        with a batch normalization layer and a relu layer.
    The corresponding kernel sizes are (1, kernel_size, 1),
        corresponding strides are (1->stride->1),
        corresponding filters are (base_depth, base_depth, depth).
    If the depth of the inputs is equal to the 'depth', this is a identity block, else a convolutional
     