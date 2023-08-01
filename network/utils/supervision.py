import copy
from math import log
import cv2
from loguru import logger

import torch
import numpy as np
from einops import repeat
from kornia.utils import create_meshgrid


@torch.no_grad()
def warp_with_grid(grid_pt0_i, sence_img_0, T_0to1, K1, device):
    sence_img_grid_0 = []
    for j in range(int(sence_img_0.shape[0])):
        sence_img_grid_0.append(torch.stack(
            [sence_img_0[j, :][i, grid_pt0_i[j, :, 1].long(), grid_pt0_i[j, :, 0].long()] for i in
             range(int(sence_img_0.shape[1]))], dim=0))
    sence_img_grid_0 = torch.stack(sence_img_grid_0, dim=0)

    sence_img_grid_0_nonmask = torch.ones(sence_img_grid_0.shape, device=device)
    sence_img_grid_0_nonmask[0, torch.where(sence_img_grid_0[0] == 0)[0], torch.where(sence_img_grid_0[0] == 0)[1]] = 0

    sence_img_grid_0_warped = T_0to1[:, :3, :3] @ sence_img_grid_0 + sence_img_grid_0_nonmask * T_0to1[:, :3, [3]]
    sence_img_grid_0_warped_with_K = (K1.float() @ sence_img_grid_0_warped).transpose(2, 1)
    sence_img_grid_0_warped_with_K_2d = sence_img_grid_0_warped_with_K[:, :, :2] / (
            sence_img_grid_0_warped_with_K[:, :, [2]] + 1e-4)

    return sence_img_grid_0_warped_with_K_2d


@torch.no_grad()
def compute_supervision_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['MODULE']['RESOLUTION'][0]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # create kpts in meshgrid and resize them to image resolution为图像生成坐标网格
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
    grid_pt0_i = scale * grid_pt0_c

    warp_with_grid_0 = warp_with_grid(grid_pt0_i, data['sence_image_0'], data["T_0to1"], data["K1"], device)

    w_pt0_c = warp_with_grid_0 / scale
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0

    conf_matrix_gt = torch.zeros(1, h0 * w0, h0 * w0, device=device)
    b_ids, i_ids = torch.where(nearest_index1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({'spv_b_ids': b_ids,
                 'spv_i_ids': i_ids,
                 'spv_j_ids': j_ids,
                 'spv_w_pt0_i': warp_with_grid_0,
                 'spv_pt1_i': grid_pt0_i})

##############  ↓  Fine-Level supervision  ↓  ##############
@torch.no_grad()
def compute_supervision_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i = data["spv_w_pt0_i"]
    pt1_i = data["spv_pt1_i"]
    scale = config['MODULE']['RESOLUTION'][1]
    radius = config['MODULE']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius

    data.update({"expec_f_gt": expec_f_gt})
