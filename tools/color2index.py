import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append('.')

from core.configures import COLOR_MAP, NAME_MAP, color2index_config
from core.utils.vis_utils import plot_image_label
from core.utils.data_utils.image_io_utils import load_image, save_to_image
from core.utils.data_utils.l