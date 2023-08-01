import numpy as np
import cv2
import os
import json
import random
import matplotlib.pyplot as plt
import time

import torch
from kornia.utils import create_meshgrid
import matplotlib.cm as cm

# from src.datasets.scared import ScaredDataset
from src.datasets.scared_new2 import ScaredDataset
from src.datasets.endoslam import EndoDataset

from Loftr.src.loftr import LoFTR
from evaluation.config.cvpr_ds_config import loftr_default_cfg

from src.network.net import net
from evaluation.config.net_v1_config import net_v1_config
from evaluation.config.net_v2_config import net_v2_config
from evaluation.config.net_v3_config import net_v3_config

from src.utils.metrics import *
from src.utils.comm import *

device = 'cuda'

