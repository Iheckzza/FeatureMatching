import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.einops import rearrange

INF = 1e9


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        self.temperature = config['dsmax_temperature']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """

        # normalize
        feat_c0_n, feat_c1_n = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc, nsc->nls", feat_c0_n, feat_c1_n) / self.temperature

        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {'h0c': data['hw0_c'][0],
                        'w0c': data['hw0_c'][1],
                        'h1c': data['hw1_c'][0],
                        'w1c': data['hw1_c'][1]}
        _device = conf_matrix.device

        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        mask_border(mask, self.border_rm, False)
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        # 2. mutual nearest
        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (
                conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        b_ids_pred, i_ids_pred, j_ids_pred = torch.where(mask == True)
        mconf = conf_matrix[b_ids_pred, i_ids_pred, j_ids_pred]

        # 4. Sampling training samples for fine-level
        if self.training:
            num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))

            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids_pred)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,),
                                             device=_device)

            conf_matrix_new = torch.zeros(conf_matrix.shape, device=_device)
            conf_matrix_new[b_ids_pred, i_ids_pred, j_ids_pred] = 1

            gt_pad_indices = torch.randint(len(data['spv_b_ids']),
                                           (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                                           device=_device)

            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids_pred, data['spv_b_ids']], [i_ids_pred, data['spv_i_ids']],
                     [j_ids_pred, data['spv_j_ids']], [mconf, mconf_gt]))
        else:
            b_ids = b_ids_pred
            i_ids = i_ids_pred
            j_ids = j_ids_pred

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='floor')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='floor')],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({'gt_mask': mconf == 0,
                               'm_bids': b_ids,
                               'mkpts0_c': mkpts0_c,
                               'mkpts1_c': mkpts1_c,
                               'mconf': mconf[mconf != 0]})

        return coarse_matches
