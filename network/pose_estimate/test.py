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

from src.network.pose_estimate.origin_match_net.origin_net import origin_net
from src.network.pose_estimate.origin_match_net.origin_match_config import origin_match_config

from src.utils.metrics import *
from src.utils.comm import *

device = 'cuda'


def make_matching_plot_fast_merge(image0, image1, kpts0, kpts1, margin=10, input_img=None, color_set=None):
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1 + margin

    if input_img is None:
        out = 255 * np.ones((H, W), np.uint8)
        out[:H0, :W0] = image0
        out[:H1, W0 + margin:] = image1
        out = np.stack([out] * 3, -1)
    else:
        out = input_img

    if color_set is None:
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    else:
        color = color_set

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    i = 0
    for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
        if i % 1 == 0:
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, color, -1, lineType=cv2.LINE_AA)
            i += 1
        else:
            i += 1

    return out


# def load_pretrained_weight(pretrained_weight):


def main():
    data_enhance = [5]
    load_data_root = r'E:\Data\processed\dataset_1\keyframe_1'
    datatset = ScaredDataset(load_data_root, mode='train', data_enhance=data_enhance, data_type='single',
                             img_size=(640, 480), read_img_gray=True)

    weight_dir = r'G:\PythonProject\TransConv\logs\transformer_versions\03.16\checkpoints\last.ckpt'
    # weight_dir = r'G:\PythonProject\TransConv\logs\transformer_versions\03.17\checkpoints\last.ckpt'
    # weight_dir = r'G:\PythonProject\TransConv\logs\transformer_versions\03.31\checkpoints\last.ckpt'
    matcher = origin_net(config=origin_match_config)

    matcher.load_state_dict(torch.load(weight_dir, map_location='cuda:0')['state_dict'])
    matcher = matcher.eval().to(device=device)

    for frame_id, data in enumerate(datatset):
        print(data['pair_names'])

        data['image0'] = data['image0'][None].to(device)
        data['image1'] = data['image1'][None].to(device)

        with torch.no_grad():
            matcher(data)

        mkpts0_f = data['mkpts0_f'][:, :2]
        mkpts1_f = data['mkpts1_f'][:, :2]


if __name__ == "__main__":
    main()
