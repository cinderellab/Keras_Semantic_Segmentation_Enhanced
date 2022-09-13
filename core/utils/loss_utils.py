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
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------
def sparse_lovasz_softmax(labels, probas, classes='all', per_image=False, ignore=0, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='all'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf