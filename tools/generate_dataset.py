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
        label = load_image(label_path, is_gray=label_is_gray, use_gdal=use_gdal)
        image_height, image_width, _ = image.shape
        image_tag = os.path.basename(image_path).split('.')[0]
        print('%s: sampling from [%s]...' % (datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'), image_path))
        # if the source image/label is too small, pad it with zeros
        if image_height < img_h:
            image = np.pad(image, ((0, img_h-image_height+1), (0, 0), (0, 0)), mode='constan