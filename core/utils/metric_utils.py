import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support


def compute_accuracy(y_true, y_pred, n_class):
    """ compute accuracy for each class and the "macro"&"micro" average accuracies.
    :param y_true: 1-D array or 