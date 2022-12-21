import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append('.')

from core.configures import COLOR_MAP, NAME_MAP, color2index_config
from core.utils.vis_utils import plot_image_label
from core.utils.data_utils.image_io_utils import load_image, save_to_image
from core.utils.data_utils.label_transform_utils import color_to_index, index_to_color


def convert_color_to_index(src_path, color_mapping, src_color_mode='rgb', dst_path=None, plot=False, names=None):
    """ convert a colorful label image to a gray (1-channel) image
        (positive index from 1~n, 0 represents background.
        If there is no background classes, there will still be 0 values)
    :param src_path: string
        path of source label image, rgb/gray color mode
    :param dst_path: string
  