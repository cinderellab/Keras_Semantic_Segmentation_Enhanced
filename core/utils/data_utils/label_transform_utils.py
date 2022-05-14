import numpy as np

def color_to_index(color_array, color_mapping, to_sparse=True):
    """ convert colorful label array to label index
        :param color_array: array of shape (height, width) or (height, width, 