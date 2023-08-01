import copy
import cv2

import torch
import numpy as np
import kornia


@torch.no_grad()
def remove_overlap_kps(origin_kp0, origin_kp1):
    overlap_kp0 = torch.div(origin_kp0, 8, rounding_mode='floor')
    overlap_kp1 = torch.div(origin_kp1, 8, rounding_mode='floor')

    overlapkp_total = torch.concat([overlap_kp0, overlap_kp1], dim=1)
    orikp_total = torch.concat([origin_kp0, origin_kp1], dim=1)

    _, idx0 = np.unique(np.array(overlapkp_total.data.cpu())[:, 2:], axis=0, return_index=True)
    idx0 = torch.tensor(idx0, dtype=torch.long)
    overlapkp_cache = overlapkp_total[idx0]
    orikp_cache = orikp_total[idx0]

    _, idx1 = np.unique(np.array(overlapkp_cache.data.cpu())[:, 2:], axis=0, return_index=True)
    idx1 = torch.tensor(idx1, dtype=torch.long)

    overlapkp_final = overlapkp_cache[idx1]
    orikp_final = orikp_cache[idx1]

    return overlapkp_final, orikp_final


@torch.no_grad()
def data_preprocess(data):
    _device = data['origin_kp0'].device
    h, w = data['image0'].shape[-2], data['image0'].shape[-1]
    h_c, w_c = h // 8, w // 8

    origin_kp0 = data['origin_kp0'][0]
    origin_kp1 = data['origin_kp1'][0]

    overlap_kp, grid_kp = remove_overlap_kps(origin_kp0, origin_kp1)
    coarse_kp0 = overlap_kp[:, :2] * 8
    coarse_kp1 = overlap_kp[:, 2:] * 8
    fine_kp0 = grid_kp[:, :2]
    fine_kp1 = grid_kp[:, 2:]

    lists_f0 = (overlap_kp[:, :2][:, 0] + w_c * overlap_kp[:, :2][:, 1]).clone().detach()
    lists_f1 = (overlap_kp[:, 2:][:, 0] + w_c * overlap_kp[:, 2:][:, 1]).clone().detach()

    fine_mtx_0 = torch.zeros((h_c * w_c, 2), dtype=torch.float32, device=_device)
    fine_mtx_1 = torch.zeros((h_c * w_c, 2), dtype=torch.float32, device=_device)

    fine_mtx_0[lists_f0.long()] = fine_kp0
    fine_mtx_1[lists_f1.long()] = fine_kp1

    data.update({'origin_kp0': origin_kp0[None]})
    data.update({'origin_kp1': origin_kp1[None]})
    data.update({'coarse_kp0': coarse_kp0[None]})
    data.update({'coarse_kp1': coarse_kp1[None]})
    data.update({'fine_kp0': fine_kp0[None]})
    data.update({'fine_kp1': fine_kp1[None]})
    data.update({'fine_mtx_0': fine_mtx_0[None]})
    data.update({'fine_mtx_1': fine_mtx_1[None]})
    data.update({'lists_f0': lists_f0[None]})
    data.update({'lists_f1': lists_f1[None]})
