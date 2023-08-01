import math
import torch
import torch.nn as nn
from loguru import logger

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from network.utils.position_encoding import PositionEncodingSine
import torch.nn.functional as F


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.mix_feat_0 = nn.Linear(49, 1, bias=True)
        self.mix_feat_1 = nn.Linear(49, 1, bias=True)
        self.pos_encoding = PositionEncodingSine(config['d_model'])

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        feat_f0_mixed = self.mix_feat_0(feat_f0.permute(0, 2, 1).contiguous())
        feat_f1_mixed = self.mix_feat_1(feat_f1.permute(0, 2, 1).contiguous())

        feat_f0_mixed = feat_f0_mixed.permute(0, 2, 1).contiguous()
        feat_f1_mixed = feat_f1_mixed.permute(0, 2, 1).contiguous()

        sim_matrix_0 = torch.einsum('mc,mrc->mr', feat_f0_mixed.squeeze(1), feat_f1)
        sim_matrix_1 = torch.einsum('mc,mrc->mr', feat_f1_mixed.squeeze(1), feat_f0)
        softmax_temp = 1. / C ** .5
        heatmap_0 = torch.softmax(softmax_temp * sim_matrix_0, dim=1).view(-1, W, W)
        heatmap_1 = torch.softmax(softmax_temp * sim_matrix_1, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized_0 = dsnt.spatial_expectation2d(heatmap_0[None], True)[0]  # [M, 2]
        coords_normalized_1 = dsnt.spatial_expectation2d(heatmap_1[None], True)[0]

        grid_normalized = create_meshgrid(W, W, True, heatmap_0.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var_0 = torch.sum(grid_normalized ** 2 * heatmap_0.view(-1, WW, 1), dim=1) - coords_normalized_0 ** 2  # [M, 2]
        std_0 = torch.sum(torch.sqrt(torch.clamp(var_0, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        var_1 = torch.sum(grid_normalized ** 2 * heatmap_1.view(-1, WW, 1), dim=1) - coords_normalized_1 ** 2  # [M, 2]
        std_1 = torch.sum(torch.sqrt(torch.clamp(var_1, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        mkpts0_f = data['mkpts0_c'] + (coords_normalized_0 * (W // 2) * self.scale + W // 2)
        mkpts1_f = data['mkpts1_c'] + (coords_normalized_1 * (W // 2) * self.scale + W // 2)

        data.update({"mkpts0_f": torch.cat([mkpts0_f, std_0.unsqueeze(1)], -1)})
        data.update({"mkpts1_f": torch.cat([mkpts1_f, std_1.unsqueeze(1)], -1)})
