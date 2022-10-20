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
    preds_fnames = os.listdir(evaluating_config.preds_dir)
    label_fnames = os.listdir(evaluating_config.label_dir)
    n_class = len(NAME_MAP[evaluating_config.dataset_name])

    if evaluating_config.mode == "global":
        mat = np.zeros((n_class, n_class))

        for preds_fname, label_fname in tqdm(zip(preds_fnames, label_fnames)):
            print(preds_fname, label_fname)
            preds = load_image(os.path.join(evaluating_config.preds_dir, preds_fname), is_gray=True)
            h, w, _ = preds.shape
            label = load_image(os.path.join(evaluating_config.label_dir, label_fname), is_gray=True, target_size=(h, w))
            _mat = confusion_matrix(label.reshape(-1), preds.reshape(-1), labels=np.arange(n_class))
            mat = mat + _mat
        if evaluating_config.ignore_0:
           