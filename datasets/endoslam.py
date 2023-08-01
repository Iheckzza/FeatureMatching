from os import path as osp

from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt

import random
import cv2
import numpy as np
from unicodedata import name

import os
import torch
import torch.utils as utils
import json
import kornia
from scipy.spatial.transform import Rotation as R


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


def matches_split_list(images_lists, pose_data_lists, keypoints_lists, data_enhance, lighting_data, scale):
    mulit_frames_data = []
    images_lists = list(images_lists)
    pose_data_lists = list(pose_data_lists)
    keypoints_lists = list(keypoints_lists)

    for ii, kk, mm in zip(images_lists, pose_data_lists, keypoints_lists):
        single_data = {'name': ii, 'pose_data': kk, 'kp_data': mm}
        mulit_frames_data.append(single_data)

    matches_final = []
    for ii in data_enhance:
        for i in range(len(mulit_frames_data) - 1):
            if lighting_data:
                k0 = i * ii
                k1 = ii * (i + 1)
                scale = scale
            else:
                k0 = i
                k1 = i + int(ii)
                scale = 1
            if k1 < len(mulit_frames_data) and k0 % scale == 0:
                matches_final.append({'img0': mulit_frames_data[k0], 'img1': mulit_frames_data[k1]})

    return matches_final


def load_all_keyframe(keyframe_root, data_enhance, lighting_data, debug_flag):
    frame_dir = os.path.join(keyframe_root, 'Frames')
    pose_dir = os.path.join(keyframe_root, 'Poses')
    kp_dir = os.path.join(keyframe_root, 'keypoints')

    frames_dir_list = [os.path.join(frame_dir, frame_list) for frame_list in sorted(os.listdir(frame_dir))]
    pose_dir = os.path.join(pose_dir, os.listdir(pose_dir)[0])

    pose_data = pd.read_excel(pose_dir).values
    poes_data_list = pose_data[:, 3:]

    keypoints_lists = [os.path.join(kp_dir, kp_list) for kp_list in sorted(os.listdir(kp_dir))]

    scale = int(np.log(len(frames_dir_list)))

    matches_final = matches_split_list(frames_dir_list, poes_data_list, keypoints_lists, data_enhance,
                                       lighting_data, scale)

    if debug_flag:
        matches_final = matches_final[:1]

    return matches_final


def cal_matches_kps(image_0, image_1, device):
    timg_gray = torch.cat([image_0[None], image_1[None]], dim=0).to(device)
    n_features = 4000
    PS = 41
    detector = kornia.feature.ScaleSpaceDetector(n_features,
                                                 resp_module=kornia.feature.BlobDoG(),
                                                 scale_space_response=True,
                                                 nms_module=kornia.geometry.ConvQuadInterp3d(10),
                                                 scale_pyr_module=kornia.geometry.ScalePyramid(3, 1.6, PS,
                                                                                               double_image=True),
                                                 ori_module=kornia.feature.LAFOrienter(19),
                                                 mr_size=6.0,
                                                 minima_are_also_good=True).to(device)
    descriptor = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)

    with torch.no_grad():
        lafs, resps = detector(timg_gray)
        patches = kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS=PS).to(device)
        B, N, CH, H, W = patches.size()
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
        scores, matches = kornia.feature.match_mnn(descs[0], descs[1])

    src_pts = lafs[0, matches[:, 0], :, 2].data.cpu().numpy()
    dst_pts = lafs[1, matches[:, 1], :, 2].data.cpu().numpy()

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = np.array(mask, dtype=bool)

    kpts0 = src_pts[mask.squeeze()]
    kpts1 = dst_pts[mask.squeeze()]
    print(len(kpts0))
    return kpts0, kpts1, H


class EndoDataset(utils.data.Dataset):
    def __init__(self, keyframe_root, mode=None, data_enhance=None, lighting_data=False,
                 read_img_gray=True, debug_flag=False, train_flag=False):
        super().__init__()

        self.keyframe_root = keyframe_root
        self.mode = mode
        self.read_img_gray = read_img_gray
        self.data_enhance = data_enhance
        self.lighting_data = lighting_data
        self.intrinsic = np.array([[957.411, 0, 282.192],
                                   [0, 959.386, 170.731],
                                   [0, 0, 1]], dtype=np.float32)

        self.distcoff = np.float32([0.2533, -0.2085, 0, 0])

        self.debug_flag = debug_flag
        self.matches_final = load_all_keyframe(self.keyframe_root, self.data_enhance,
                                               self.lighting_data, self.debug_flag)

        self.train_flag = train_flag
        self.device = kornia.utils.helpers.get_cuda_device_if_available()

    def __len__(self):
        return len(self.matches_final)

    def _compute_rel_pose(self, pose0, pose1):
        return np.linalg.inv(np.matmul(np.linalg.inv(pose1), pose0))

    def __getitem__(self, idx):

        selected_matches = self.matches_final[idx]

        image_0_dict = selected_matches['img0']
        image_1_dict = selected_matches['img1']

        kp_data_dict0 = np.load(image_0_dict['kp_data'], allow_pickle=True)
        kp_data_dict1 = np.load(image_1_dict['kp_data'], allow_pickle=True)
        lafs0, descs0 = torch.tensor(kp_data_dict0.item()['lafs'], device=self.device), torch.tensor(
            kp_data_dict0.item()['descriptor'], device=self.device)
        lafs1, descs1 = torch.tensor(kp_data_dict1.item()['lafs'], device=self.device), torch.tensor(
            kp_data_dict1.item()['descriptor'], device=self.device)

        scores, matches = kornia.feature.match_mnn(descs0, descs1)
        src_pts = lafs0[matches[:, 0], :, 2]
        dst_pts = lafs1[matches[:, 1], :, 2]

        # ransac = kornia.geometry.RANSAC(model_type='homography').to(self.device)
        # _, mask = ransac(src_pts, dst_pts)
        H, mask = cv2.findHomography(np.array(src_pts.data.cpu()), np.array(dst_pts.data.cpu()), cv2.RANSAC, 5.0)
        mask = torch.tensor(mask, dtype=torch.bool)

        origin_kp0 = src_pts[mask.squeeze()]
        origin_kp1 = dst_pts[mask.squeeze()]

        if self.train_flag:
            if len(origin_kp0) < 200 and idx < len(self.matches_final) - 1:
                torch.cuda.empty_cache()
                return self.__getitem__(idx + 1)

        if self.read_img_gray:
            image_0_origin = cv2.imread(image_0_dict['name'], cv2.IMREAD_GRAYSCALE)
            image_1_origin = cv2.imread(image_1_dict['name'], cv2.IMREAD_GRAYSCALE)

            image_0 = torch.from_numpy(image_0_origin).float()[None] / 255
            image_1 = torch.from_numpy(image_1_origin).float()[None] / 255
        else:
            image_0_origin = cv2.imread(image_0_dict['name'], -1)
            image_1_origin = cv2.imread(image_1_dict['name'], -1)

            image_0 = (torch.from_numpy(image_0_origin).float() / 255).permute(2, 0, 1).contiguous()
            image_1 = (torch.from_numpy(image_1_origin).float() / 255).permute(2, 0, 1).contiguous()

        pose_data_0 = image_0_dict['pose_data']
        pose_data_1 = image_1_dict['pose_data']

        rotate_0 = R.from_quat(pose_data_0[3:]).as_matrix()
        rotate_1 = R.from_quat(pose_data_1[3:]).as_matrix()
        t_0 = pose_data_0[:3][None]
        t_1 = pose_data_1[:3][None]

        pose_0 = np.eye(4, dtype=np.float32)
        pose_1 = np.eye(4, dtype=np.float32)
        pose_0[:3, :3] = rotate_0
        pose_1[:3, :3] = rotate_1
        pose_0[:3, 3:] = t_0.T
        pose_1[:3, 3:] = t_1.T

        Rt_arms = np.array([[0.9463, -0.0921, -0.3098, -46.2017],
                            [-0.1389, 0.7495, -0.6472, 20.9074],
                            [0.2918, -0.6555, 0.8965, 94.6349],
                            [0, 0, 0, 1]])

        # pose_0 = np.dot(pose_0, Rt_arm)

        T_0to1 = torch.tensor(self._compute_rel_pose(pose_0, pose_1), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        # device = kornia.utils.helpers.get_cuda_device_if_available()
        # kpts0c, kpts1c, H = cal_matches_kps(image_0, image_1, device)
        # matches = make_matching_plot_fast_merge(image_0_origin, image_1_origin, kpts0c, kpts1c)
        # cv2.imshow("image_0_origin", image_0_origin)
        # cv2.imshow("image_1_origin", image_1_origin)
        # cv2.imshow('matches', matches)
        # cv2.waitKey(1)

        kp_data_dict0 = np.load(image_0_dict['kp_data'], allow_pickle=True)
        kp_data_dict1 = np.load(image_1_dict['kp_data'], allow_pickle=True)

        lafs0, descs0 = kp_data_dict0.item()['lafs'], kp_data_dict0.item()['descriptor']
        lafs1, descs1 = kp_data_dict1.item()['lafs'], kp_data_dict1.item()['descriptor']

        data = {
            'image0': image_0,  # (1, h, w)
            'image1': image_1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': self.intrinsic,  # (3, 3)
            'K1': self.intrinsic,
            'pair_id': idx,
            'pair_names': (image_0_dict['name'],
                           image_1_dict['name']),
            'lafs0': lafs0,
            'lafs1': lafs1,
            'descs0': descs0,
            'descs1': descs1,
            'origin_kp0': origin_kp0,
            'origin_kp1': origin_kp1,
        }
        return data


if __name__ == '__main__':
    data_root = r'E:\Data\EndoSLAM\Cameras\HighCam\test\Small Intestine'
    testclass = EndoDataset(data_root, mode='train', data_enhance=[5],
                            lighting_data=False, read_img_gray=True,train_flag=False)

    print(len(testclass))

    for test in testclass:
        print(test['pair_names'])
        # descs0, descs1 = torch.tensor(test['descs0'], device='cuda'), torch.tensor(test['descs1'], device='cuda')
        # lafs0, lafs1 = test['lafs0'], test['lafs1']
        # scores, matches = kornia.feature.match_mnn(descs0, descs1)
        #
        # src_pts = lafs0[matches[:, 0].data.cpu(), :, 2]
        # dst_pts = lafs1[matches[:, 1].data.cpu(), :, 2]
        #
        # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # mask = torch.tensor(mask, dtype=torch.bool)
        #
        # origin_kp0 = src_pts[mask.squeeze()]
        # origin_kp1 = dst_pts[mask.squeeze()]

        image0 = np.array(test['image0'][0] * 255)
        image1 = np.array(test['image1'][0] * 255)

        origin_kp0 = np.array(test['origin_kp0'].data.cpu())
        origin_kp1 = np.array(test['origin_kp1'].data.cpu())

        img = make_matching_plot_fast_merge(image0, image1, origin_kp0, origin_kp1, color_set=[0, 0, 230])

        print(origin_kp0.shape[0])
        cv2.imshow('img', img)
        cv2.waitKey(1)
