import torch
import torch.nn as nn
import math
import numpy as np
import cv2
import torch.nn.functional as F

from einops.einops import rearrange
from Loftr.src.loftr_module import LocalFeatureTransformer

from Loftr.src.cvpr_ds_config import loftr_default_cfg

INF = 1e9


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, block=BasicBlock, initial_dim=128, block_dims=[128, 196, 256]):
        super().__init__()
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x0 = self.relu(self.bn1(self.conv1(x)))
            x1 = self.layer1(x0)  # 1/2
            x2 = self.layer2(x1)  # 1/4
            x3 = self.layer3(x2)  # 1/8

            # FPN
            x3_out = self.layer3_outconv(x3)

            x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
            x2_out = self.layer2_outconv(x2)
            x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

            x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
            x1_out = self.layer1_outconv(x1)
            x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return [x3_out, x1_out]


class PositionEncodingSine(nn.Module):
    def __init__(self, d_model=256, max_shape=(256, 256)):
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


#######################################################################################

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
    def __init__(self):
        super().__init__()
        self.thr = 0.2
        self.border_rm = 2

        # we provide 2 options for differentiable matching
        self.match_type = 'dual_softmax'

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / 0.1

        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    def get_coarse_match(self, conf_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
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
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

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

        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches


class LoFTR_tiny(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = ResNetFPN_8_2()
        self.pos_encoding = PositionEncodingSine()
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching()

    def forward(self, data):
        with torch.no_grad():
            data.update({
                'bs': data['image0'].size(0),
                'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
            })

            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])

            data.update({'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
                         'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

            # 2. coarse-level loftr module
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
            feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

            mask_c0 = mask_c1 = None
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

            # 3. match coarse-level
            self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


if __name__ == "__main__":
    img0 = cv2.imread(r'E:\Data\processed\dataset_1\keyframe_1\raw_images\Left\000000.png', 0)
    img1 = cv2.imread(r'E:\Data\processed\dataset_1\keyframe_1\raw_images\Left\000010.png', 0)
    # img0 = cv2.imread(
    #     r'/home/zhangziang/Porject/DATA/scared_modified/train/dataset_1/keyframe_1/raw_images/Left/000000.png', -1)
    # img1 = cv2.imread(
    #     r'/home/zhangziang/Porject/DATA/scared_modified/train/dataset_1/keyframe_1/raw_images/Left/000010.png', -1)
    input0 = torch.tensor(img0[None][None], dtype=torch.float32, device='cuda')
    input1 = torch.tensor(img1[None][None], dtype=torch.float32, device='cuda')

    weight = r'G:\PythonProject\TransConv\Loftr\demo\weights\outdoor_ds.ckpt'

    batch = {'image0': input0, 'image1': input1}

    matcher = LoFTR_tiny(loftr_default_cfg).to('cuda')
    matcher.load_state_dict(torch.load(weight, map_location='cuda:0')['state_dict'], strict=False)
    matcher = matcher.eval().to('cuda')
    output = matcher(batch)
    print(output)
