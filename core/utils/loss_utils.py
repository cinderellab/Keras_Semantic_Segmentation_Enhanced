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


def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def categorical_crossentropy_seg(y_true, y_pred):
    """
    :param y_true: tensor of shape (batch_size, height, width, n_class)
    :param y_pred: tensor of shape (batch_size, height, width, n_class)
        probability predictions after softmax
    :return: categorical cross-entropy
    """
    n_class = K.int_shape(y_pred)[-1]

    y_true = K.reshape(y_true, (-1, n_class))
    y_pred = K.log(K.reshape(y_pred, (-1, n_class)))

    cross_entropy = -K.sum(y_true * y_pred, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def sparse_categorical_crossentropy_seg(y_true, y_pred):
    """ calculate cross-entropy of the one-hot prediction and the sparse gt.
    :param y_true: tensor of shape (batch_size, height, width)
    :param y_pred: tensor of shape (batch_size, height, width, n_class)
    :return: categorical cross-entropy
    """
    n_class = K.int_shape(y_pred)[-1]

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), n_class)
    y_pred = K.log(K.reshape(y_pred, (-1, n_class)))

    cross_entropy = -K.sum(y_true * y_pred, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    B