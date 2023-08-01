import math
import torch
import torch.nn as nn
import numpy as np
from einops.einops import rearrange

import matplotlib.pyplot as plt
from network.backbone import build_backbone

from network.module import LocalFeatureTransformer, FinePreprocess

from network.utils.position_encoding import PositionEncodingSine
from network.utils.coarse_matching_new import CoarseMatching
from network.utils.pose_pred import Pose_Pred
from network.utils.pose_pred_new import Pose_Pred_new

from network.utils.fine_matching_new import FineMatching


class net(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching(config["fine"])
        if self.config['pose_net_flag'] == 'old':
            self.pose_pred = Pose_Pred(config['pose'])
        elif self.config['pose_net_flag'] == 'new':
            self.pose_pred = Pose_Pred_new(config['pose_new'])
        else:
            self.pose_pred = None

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        # for swin-transformer backbone there is no need to add position encoding.
        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        # 3. match coarse-level
        mask_c0 = mask_c1 = None
        feat_c0, feat_c1 = self.coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # 6. pose predict
        if self.pose_pred is not None:
            self.pose_pred(feat_c0, feat_c1, data)

        data['feat_c0'] = feat_c0
        data['feat_c1'] = feat_c1
        data['feat_f0'] = feat_f0
        data['feat_f1'] = feat_f1

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        for j in list(state_dict.keys()):
            if j.startswith('loftr_'):
                state_dict[j.replace('loftr_', '', 1)] = state_dict.pop(j)

        return super().load_state_dict(state_dict, *args, **kwargs)


if __name__ == '__main__':
    from src.network.configs.config_pred_mdified import cfg_new
    from src.datasets.scared_new2 import ScaredDataset
    matcher = net(config=cfg_new)

    load_data_root = r'E:\Data\processed\dataset_9\keyframe_1'
    datatset = ScaredDataset(load_data_root, mode='train', data_enhance=[1], data_type='single',
                             img_size=(640, 480), read_img_gray=False)

    data = datatset[0]
    data['image0'] = data['image0'][None]
    data['image1'] = data['image1'][None]

    matcher(data)

    print(matcher)



