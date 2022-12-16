import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append('.')

from core.configures import COLOR_MAP, NAME_MAP, color2index_config
from cor