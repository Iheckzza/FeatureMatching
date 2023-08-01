import copy
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
import json

import torch
from kornia.utils import create_meshgrid
from datasets.scared_new2 import ScaredDataset
from datasets.endoslam import EndoDataset

from network.net import net
from net_config import net_configs

torch.set_grad_enabled(False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def draw_matches(img1, img2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    out[:rows1, :cols1] = np.dstack([img1, img1, img1])
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])

    count = 0
    for mat in matches:

        (x1, y1) = mat[0]
        (x2, y2) = mat[1]

        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)

        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (a, b, c), 1)
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (a, b, c), 1)

        if count == 1:
            cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))),
                     (a, b, c),
                     1, shift=0)
            count = 0
        else:
            count += 1
            continue

    return out


def make_matching_plot_fast_merge(image0, image1, kpts0, kpts1, inliers_gt=None, margin=10, input_img=None,
                                  color_set=None):
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1 + margin

    if input_img is None:
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0 + margin:, :] = image1
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
            color = [0, 200, 0]
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=color, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, color, -1, lineType=cv2.LINE_AA)
            i += 1
        else:
            i += 1

    return out


def main_demo():
    device = 'cuda'
    image_0 = cv2.imread(r'.png', -1)  ##path to image_0
    image_1 = cv2.imread(r'.png', -1)  ##path to imahe_1
    weight = '.weights/last.ckpt'

    data = {}
    data['image0'] = torch.tensor(image_0, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
    data['image1'] = torch.tensor(image_1, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    matcher = net(config=net_configs)
    matcher.load_state_dict(torch.load(weight, map_location='cuda:0')['state_dict'])
    matcher = matcher.eval().to(device='cuda')
    matcher(data)

    mkpts0 = data['mkpts0_f'].cpu().numpy()
    mkpts1 = data['mkpts1_f'].cpu().numpy()

    match_image = make_matching_plot_fast_merge(image_0, image_1, mkpts0[:, :2], mkpts1[:, :2], color_set=[0, 200, 0],
                                                input_img=None)
    cv2.imshow('match_image', match_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main_demo()
