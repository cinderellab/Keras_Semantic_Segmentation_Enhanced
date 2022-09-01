import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import array_ops


# https://blog.csdn.net/wangdongwei0/article/details/84576044
def log_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


# TODO: TO BE TESTED
# binary dice loss
def _dice_coef_binary(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss_binary(y_true, y_pred):
    return 1 - _dice_coef_binary(y_true, y_pred, smooth=1)


# y_true and y_pred should be one-hot
# y_true.shape = (None,Width,Height,Channel)
# y_pred.shape = (None,Width,Height,Channel)
def _dice_coef_multiclass(y_true, y_pred, smooth=1):
    mean_loss = 0
    for i in range(y_pred.shape(-1)):
        intersection = K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
        union = K.sum(y_true[:,:,:,i], axis=[1,2,3]) + K.sum(y_pred[:,:,:,i], axis=[1,2,3])
        mean_loss += (2. * intersection + smooth) / (union + smooth)
    return K.mean(mean_loss, axis=0)


def dice_coef_loss_multiclass(y_true, y_pred):
    return 1 - _dice_coef_multiclass(y_true, y_pred, smooth=1)


def focal_