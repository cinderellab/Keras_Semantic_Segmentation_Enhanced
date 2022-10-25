
import os
import datetime
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import sys
sys.path.append('.')

from core.utils.data_utils.image_io_utils import load_image, get_image_info, save_to_image_gdal, save_to_image
from core.utils.vis_utils import plot_segmentation
from core.configures import COLOR_MAP, NAME_MAP, predicting_config, net_config
from core.nets import SemanticSegmentationModel


def predict_stride(model, image_path, patch_height=256, patch_width=256, stride=None, to_prob=False, plot=False,
                   geo=False, dataset_name="voc"):
    """ predict labels of a large image, usually for remote sensing HSR tiles
        or for images that size are not consistent with the required input size.
    :param model: Keras Model instance
    :param image_path: string, path of input image.
    :param patch_height: int, default 256.
    :param patch_width: int, default 256.
    :param stride: int, default None.
    :param to_prob: bool, whether to return probability.
    :param plot: bool, whether to plot.
    :param geo: bool, whether to load geo images.
    :param: bool, whether to plot.
    :param dataset_name: string.