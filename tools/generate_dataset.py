import datetime
import numpy as np
import os

import sys
sys.path.append('.')

from core.utils.data_utils.image_io_utils import load_image, save_to_image, save_to_image_gdal
from core.configures import generate_dadaset_config


def generate_dataset_random(image_paths,
                            label_paths,
                            dst_dir = './training',
                            image_num_per_tile=10,
                            img_h=256,
                            img_w=256,
                            label_is_gray=True,
                            use_gdal=False):
    # Assuming that the source images are common images with 3 bands, and the label images are images with 1 or 3 bands.
    # check source directories and create directories to store sample images and gts
    if not os.path.exists('{}/image'.format(dst_dir)):
        os.mkdir('{}/image'.format(dst_dir))
    if not os.path.exists('{}/label'.format(dst_dir)):
        os.mkdir('{}/label'.format(dst_dir))

    # number of samples for each image
    for image_path, label_path in zip(image_paths, label_paths):
        image = load_image(image_path, is_gray=False, use_gdal=use_gdal)
        label = load_image(label_path, is_gray=label_is_gray, use_