import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('.')

from core.configures import NAME_MAP, evaluating_config
from core.utils.data_utils.image_io_utils import load_image
from core.utils.metric_utils import compute_global_metrics, compute_metrics_per_image


def evaluating_main():
    preds_fnames = os.listdir(evaluating_config.pred