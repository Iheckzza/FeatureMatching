import copy
from math import log
import cv2
from loguru import logger

import torch
import numpy as np
from einops import repeat
from kornia.utils import create_meshgrid


@torch.no_grad()
def compute_supervision_coarse(data, config):
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['MODULE']['RESOLUTION'][0]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    coarse_kp0 = data['coarse_kp0']
    coarse_kp1 = data['coarse_kp1']

    coarsescale_kp0 = coarse_kp0 / 8
    coarsescale_kp1 = coarse_kp1 / 8

    i_ids = coarsescale_kp0[..., 0] + coarsescale_kp0[..., 1] * w0
    j_ids = coarsescale_kp1[..., 0] + coarsescale_kp1[..., 1] * w1

    b_ids = torch.zeros(i_ids.shape, dtype=torch.long).squeeze().to(device)
    i_ids = i_ids.squeeze().long().to(device)
    j_ids = j_ids.squeeze().long().to(device)
    conf_matrix_gt = torch.zeros(1, h0 * w0, h0 * w0, device=device)
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    data.update({'conf_matrix_gt': conf_matrix_gt})

    if len(i_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({'spv_b_ids': b_ids,
                 'spv_i_ids': i_ids,
                 'spv_j_ids': j_ids,
                 'spv_fine_0': data['fine_kp0'],
                 'spv_fine_1': data['fine_kp1']})

@torch.no_grad()
def compute_supervision_fine(data):

    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    fine_mtx_0 = data['fine_mtx_0']
    fine_mtx_1 = data['fine_mtx_1']

    data.update({"expec_f_gt_0": fine_mtx_0[b_ids, i_ids],
                 "expec_f_gt_1": fine_mtx_1[b_ids, j_ids]})
