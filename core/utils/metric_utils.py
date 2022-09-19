import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support


def compute_accuracy(y_true, y_pred, n_class):
    """ compute accuracy for each class and the "macro"&"micro" average accuracies.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :return:
        metrics: array, shape=[n_class].
        avg_macro_metric: float, macro average accuracy.
        avg_micro_metric: float, micro average accuracy.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    labels = np.asarray([i for i in range(n_class)])
    _mat = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = np.zeros(n_class)

    for i in range(n_class):
        t_count = np.sum(_mat, axis=1)[i]
        metrics[i] = np.nan if t_count == 0 else _mat[i][i] / t_count

    avg_macro_metric = np.nanmean(metrics)
    avg_micro_metric = np.sum(_mat.diagonal(offset=0, axis1=0, axis2=1)) / np.sum(_mat)
    return metrics, avg_macro_metric, avg_micro_metric


def compute_precision_recall_f1(y_true, y_pred, n_class, avg="weighted"):
    """ compute the precision, recall and f1-score.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :param avg: string, average methods, one of ["micro", "macro", "weighted"].

    :return:
        precision_metrics: array, shape=[n_class].
        recall_metrics: array, sh