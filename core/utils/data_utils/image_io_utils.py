"""
utilization for load/save images, note that all the image are loaded to a [3-dim] array.
"""
import cv2
from PIL import Image
import os
import osr
from osgeo import gdal
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def _load_image_gdal(image_path, value_scale=1.0):
    """ using gdal to read image, especially for remote sensing multi-spectral images
    :param image_p