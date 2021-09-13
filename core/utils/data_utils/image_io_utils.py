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
    :param image_path: string, image path
    :param value_scale: float, default 1.0. the data array will divided by the 'value_scale'

    :return: array of shape (height, width, band)
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    col=ds.RasterXSize
    row=ds.RasterYSize
    band=ds.RasterCount

    img=np.zeros((row, col, band))
    for i in range(band):
        dt = ds.GetRasterBand(i+1)
        img[:,:,i] = dt.ReadAsArray(0, 0, col, row)

    return img / value_scale


def load_image(image_path, is_gray=False, value_scale=1, target_size=None, use_gdal=False):
    """ load image
    :param image_path: string, image path
    :param is_gray: bool, default False
    :param value_scale: float, default 1. the data array will divided by the 'value_scale'
    :param target_size: tuple, default None. target spatial size to resize. If None, no resize will be performed.
    :param use_gdal: bool, default False.  whether use gdal to load data,  this is usually used for loading
    multi-spectral images, or images with geo-spatial information

    :return: array of shape (height, width, band)
    """
    assert value_scale!=0
    if use_gdal:
        # if use gdal, resize and gray are valid
        return _load_image_gdal(image_path, value_scale) / value_scale
    if is_gray:
        try:
            img = img_to_array(load_img(image_path, color_mode="grayscale", target_size=target_size))
        except:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if target_size is not None:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=-1)
    else:
        try:
            img = img_to_array(load_img(image_path, target_size=target_size))
        except:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if target_size is not None:
                img = cv2.resize(img, target_size)

    return img / value_scale


def get_image_info(image_path, get_rows=False, get_cols=False, get_bands=False, get_geotransform=False, get_projection=False, get_nodatavalue=False):
    """ get the basic information of a image
    :param image_path: string
        image path
    :param get_rows: bool, default False
        whether to get rows
    :param get_cols: bool, default False
        whether