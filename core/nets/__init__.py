from .unets import UNet, ResUNet, MobileUNet
from .pspnets import PSPNet
from .deeplabs import Deeplab_v3, Deeplab_v3p
from .dense_aspp import DenseASPP
from .fcns import FCN_8s, FCN_16s, FCN_32s
from .refinenets import RefineNet
from .segnets import SegNet
from .srinets import sri_net


def SemanticSegmentationModel(model_name,
                              input_shape,
                              n_class,
                              encoder_name,
                              encoder_weights=None,
                              init_filters=64,
                              weight_decay=1e-4,
                              kernel_initializer="he_normal",
                              bn_epsilon=1e-3,
                              bn_momentum=0.99,
                              dropout=0.5,
                              upscaling_method="bilinear"):
    """ the main api of model builder.
    :param model_name: string, name of FCN model.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of classes, at least 2.
    :param encoder_name: string, name of the encoder.
    :param encoder_weights: string, path of the encoder.
    :param init_filters: int, initial filters, only used for some of the models like