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
                     