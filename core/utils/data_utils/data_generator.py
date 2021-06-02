
from keras.preprocessing.image import apply_brightness_shift, apply_channel_shift

from .image_augmentation_utils import *
from .directory_iterator import SegDirectoryIterator


class ImageDataGenerator(object):
    def __init__(self,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 zoom_range=0,
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False
                 ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.zoom_maintain_shape = zoom_maintain_shape
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.channel_axis = 2
        self.row_axis = 0
        self.col_axis = 1
        #self.ch_mean = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('Invalid zoom_range: {}. Expected to be a float or a tuple or list of two floats. '.format(zoom_range))


    def flow_from_directory(self,
                            base_fnames,
                            image_dir,
                            image_suffix,