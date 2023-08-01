import copy

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
import einops

from network.module.transformer import LocalFeatureTransformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, int(2 * dim), bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3 [rz, ry, rx]
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    z = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    x = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def transformation_from_parameters(axisangle, translation):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    R[:, :3, 3:] = t.view(-1, 3, 1)

    return R


def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def estimate_pose(kpts0, kpts1, K0):
    if len(kpts0) < 5:
        return None

    focal_length = 0.5 * (K0[0, 0] + K0[1, 1])
    principle_point = (K0[0, 2], K0[1, 2])
    E, mask = cv2.findEssentialMat(kpts0, kpts1, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)

    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


class Pose_Pred_new(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.origin_size = config['size']
        self.axis_weight = config['axis_w']
        self.trans_weight = config['trans_w']
        self.featuretransformer = LocalFeatureTransformer(config)

        self.downsample = PatchMerging(256, norm_layer=nn.LayerNorm)
        self.norm = nn.LayerNorm(256 * 2)
        self.head = nn.Linear(256 * 2, 256)

        self.pose_conv = nn.Conv2d(256, 7, kernel_size=(1, 1))
        self.quat_layer = nn.Softmax(dim=0)

    def forward(self, feat_0, feat_1, data):
        h, w = self.origin_size[0], self.origin_size[1]

        f_0, f_1 = self.featuretransformer(feat_0, feat_1)
        f_0_down = self.downsample(f_0, h, w)
        f_1_down = self.downsample(f_1, h, w)

        f_0_down = f_0_down.view(-1, h // 2, w // 2, f_0_down.shape[-1]).permute(0, 3, 1, 2).contiguous()
        f_1_down = f_1_down.view(-1, h // 2, w // 2, f_1_down.shape[-1]).permute(0, 3, 1, 2).contiguous()

        cat_f = torch.cat([f_0_down, f_1_down], dim=0)
        out_norm = self.norm(cat_f.permute(0, 2, 3, 1).contiguous())
        out1 = self.head(out_norm)
        out2 = self.pose_conv(out1.permute(0, 3, 1, 2).contiguous())
        out = out2.mean(3).mean(2)

        translation = out[..., :3]
        quot = out[..., 3:]
        quot_out = self.quat_layer(quot)

        data.update({"translate_pred": translation[None]})
        data.update({"quot_pred": quot_out[None]})

        T_0to1_pred, T_1to0_pred = self.calculate_rt(quot_out, translation)

        data.update({"T_0to1_pred": T_0to1_pred[None]})
        data.update({"T_1to0_pred": T_1to0_pred[None]})

    @torch.no_grad()
    def calculate_rt(self, quto, translate):
        device = quto.device
        rotate_0_pred = R.from_quat(np.array(quto[0, :].data.cpu())).as_matrix()
        rotate_1_pred = R.from_quat(np.array(quto[1, :].data.cpu())).as_matrix()

        T_0to1_pred = torch.eye(4).to(device=device)
        T_1to0_pred = torch.eye(4).to(device=device)

        T_0to1_pred[:3, :3] = torch.tensor(rotate_0_pred, device=device)
        T_0to1_pred[:3, 3:] = translate[0][None].T

        T_1to0_pred[:3, :3] = torch.tensor(rotate_1_pred, device=device)
        T_1to0_pred[:3, 3:] = translate[1][None].T

        return T_0to1_pred, T_1to0_pred
