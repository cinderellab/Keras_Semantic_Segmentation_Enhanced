import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import array_ops


# https://blog.csdn.net/wangdongwei0/article/details/84576044
def log_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


# TODO: TO BE TESTED
# binary dice loss
def _dice_coef_binary(y_true, y_pred, smooth=1):
    intersection