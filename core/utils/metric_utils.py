import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support


def compute_accuracy(y_true, y_pred, n_class):
    """ compute accuracy for each class and the "macro"&"micro" average accuracies.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :