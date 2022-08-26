import numpy as np

def color_to_index(color_array, color_mapping, to_sparse=True):
    """ convert colorful label array to label index
        :param color_array: array of shape (height, width) or (height, width, 3)
            RGB or gray label array, where the pixel value is not the label index
        :param colour_mapping: array of shape (n_class,) or (n_class, 3), where n_class is the [total] class number
            Note: the first element is the color of background (label 0).
            e.g., if colour_mapping=[0, 255], pixel equal to 255 are assigned with 1, otherwise 0
            if colour_mapping=[[0, 0, 0], [255,255,255]], pixel equal to [255, 255, 255] are assigned with 1, otherwise 0
        :param to_sparse: bool, optional, default True
            whether to apply a argmax on the last axis to obtain sparse label array

        :return: array of shape (height, width, n_class+1)
        NOTE: if a pixel value pair is not in the colour_mapping, the value of that pixel in the final one-hot array will be [0, 0, ..., 0]
    """
    assert color_mapping is not None
    if len(color_ma