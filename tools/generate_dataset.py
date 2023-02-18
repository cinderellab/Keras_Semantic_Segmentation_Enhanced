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
    # Assuming that the source images are common images