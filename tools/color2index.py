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
        path of destination label image, gray color mode, index from 0 to n (n is the number of non-background classes)
    :param src_color_mode: string, "rgb" or "gray", default "rgb"
        color mode of the source label image
    :param color_mapping: list or array, default None
        a list like [0, 255], [[1, 59, 3], [56, 0, 0]]
    :param plor: bool, default False
        whether to plot comparison
    :param names: list.

    :return: None
    """
    if color_mapping is None:
        raise ValueError('Invalid color mapping: None. Expected not None!')
    if src_color_mode=='rgb':
        label_color = load_image(src_path, is_gray=False).astype(np.uint8)
    elif src_color_mode=='gray':
        label_color = load_image(src_path, is_gray=True).astype(np.uint8)
    else:
        raise ValueError('Invalid src_color_mode: {}. Expected "rgb" or "gray"!'.format(src_color_mode))

    label_index = color_to_index(label_color, color_mapping, to_sparse=True)
    if np.max(label_index)>=len(color_mapping):
        raise ValueError('max value is large than: {}：{}'.format(len(colo