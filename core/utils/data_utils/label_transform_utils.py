import numpy as np

def color_to_index(color_array, color_mapping, to_sparse=True):
    """ convert colorful label array to label index
        :param color_array: array of shape (height, width) or (height, width, 3)
            RGB or gray label array, where the pixel value is not the label index
        :param colour_mapping: array of shape (n_class,) or (n_class, 3), where n_class is the [total] class number
            Note: the first element is the color of background (label 0).
            e.g., if colou