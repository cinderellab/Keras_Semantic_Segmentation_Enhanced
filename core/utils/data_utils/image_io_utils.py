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
        img[:,:,i] = dt.ReadAsArray(0, 0,