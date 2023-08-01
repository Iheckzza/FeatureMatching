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
        current_enhance = []
        for i in range(len(mulit_frames_data) - 1):
            k0 = i
            k1 = i + int(ii)
            if k0 < len(mulit_frames_data) and k1 < len(mulit_frames_data):
                current_enhance.append({'img0': mulit_frames_data[k0], 'img1': mulit_frames_data[k1]})
        if lighting_data:
            if int(len(mulit_frames_data) * scale) < len(current_enhance):
                matches_final.extend(random.sample(current_enhance, int(len(mulit_frames_data) * scale)))
            else:
                matches_final.extend(current_enhance)
        else:
            matches_final.extend(current_enhance)

    return matches_final


def load_all_keyframe(keyframe_root, data_enhance, lighting_data, debug_flag):
    frame_dir = os.path.join(keyframe_root, 'Frames')
    pose_dir = os.path.join(keyframe_root, 'Poses')
    depth_dir = os.path.join(keyframe_root, 'Pixelwise Depths')
    kp_dir = os.path.join(keyframe_root, 'keypoints')

    frames_dir_list = [os.path.join(frame_dir, frame_list) for frame_list in sorted(os.listdir(frame_dir))]
    pose_dir = os.path.join(pose_dir, os.listdir(pose_dir)[0])

    pose_data = pd.read_csv(pose_dir).values
    poes_data_list = pose_data[:, :-1]

    depth_list = [os.path.join(depth_dir, kp_list) for kp_list in sorted(os.listdir(depth_dir))]
    keypoints_lists = [os.path.join(kp_dir, kp_list) for kp_list in sorted(os.listdir(kp_dir))]

    scale = np.log(len(frames_dir_list)) * 0.02

    matches_final = matches_split_list(frames_dir_list, poes_data_list, keypoints_lists, data_enhance,
                                       lighting_data, scale)

    if debug_flag:
        matches_final = matches_final[:1]

    return matches_final


def cal_kps(image_0, device):
    image = image_0[None].to(device)
    n_features = 8000
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
        lafs, resps = detector(image)
        patches = kornia.feature.extract_patches_from_pyramid(image, lafs, PS=PS).to(device)
        B, N, CH, H, W = patches.size()
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

    return lafs[0], descs[0]


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


class UnityDataset(utils.data.Dataset):
    def __init__(self, keyframe_root, mode=None, data_enhance=None, lighting_data=False,
                 read_img_gray=True, debug_flag=False, train_flag=True):
        super().__init__()

        self.keyframe_root = keyframe_root
        self.mode = mode
        self.read_img_gray = read_img_gray
        self.data_enhance = data_enhance
        self.lighting_data = lighting_data
        self.intrinsic = np.array([[156.0418, 0, 178.5604],
                                   [0, 155.7529, 181.8043],
                                   [0, 0, 1]], dtype=np.float32)

        self.debug_flag = debug_flag
        self.matches_final = load_all_keyframe(self.keyframe_root, self.data_enhance,
                                               self.lighting_data, self.debug_flag)

        self.train_flag = train_flag
        self.device = kornia.utils.helpers.get_cuda_device_if_available()
        # self.disk = kornia.feature.DISK.from_pretrained('depth').to(self.device)
        self.disk = kornia.feature.DISK.from_pretrained('depth')

    def __len__(self):
        return len(self.matches_final)

    def _compute_rel_pose(self, pose0, pose1):
        return np.linalg.inv(np.matmul(np.linalg.inv(pose1), pose0))

    def remove_none_kps(self, origin_kp0, origin_kp1, h, w):
        o_locate = torch.tensor((h // 2, w // 2), device=origin_kp0.device)
        d_0 = torch.sum((origin_kp0 - o_locate) ** 2, dim=-1) ** 0.5
        d_1 = torch.sum((origin_kp1 - o_locate) ** 2, dim=-1) ** 0.5

        mask_d0 = torch.where(d_0 <= 170, 1, 0)
        mask_d1 = torch.where(d_1 <= 170, 1, 0)
        mask_ = (mask_d0 * mask_d1).bool()

        origin_kp0 = origin_kp0[mask_]
        origin_kp1 = origin_kp1[mask_]

        return origin_kp0, origin_kp1

    def __getitem__(self, idx):
        selected_matches = self.matches_final[idx]

        image_0_dict = selected_matches['img0']
        image_1_dict = selected_matches['img1']

        image_0_rgb = cv2.imread(image_0_dict['name'], -1)
        image_1_rgb = cv2.imread(image_1_dict['name'], -1)
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

        kp_data_dict0 = np.load(image_0_dict['kp_data'], allow_pickle=True)
        kp_data_dict1 = np.load(image_1_dict['kp_data'], allow_pickle=True)
        lafs0, descs0 = torch.tensor(kp_data_dict0.item()['lafs']), torch.tensor(kp_data_dict0.item()['descriptor'])
        lafs1, descs1 = torch.tensor(kp_data_dict1.item()['lafs']), torch.tensor(kp_data_dict1.item()['descriptor'])
        scores, matches = kornia.feature.match_smnn(descs0, descs1)
        origin_kp0 = lafs0[matches[:, 0], :, 2]
        origin_kp1 = lafs1[matches[:, 1], :, 2]

        # inp_disk = torch.cat([image_0[None], image_1[None]], dim=0).to(self.device)
        inp_disk = torch.cat([torch.from_numpy(image_0_rgb).permute(2, 0, 1).contiguous().float()[None],
                              torch.from_numpy(image_1_rgb).permute(2, 0, 1).contiguous().float()[None]], dim=0)
        with torch.no_grad():
            features0, features1 = self.disk(inp_disk, 2048, pad_if_not_divisible=True)
        kps0, descs0_d = features0.keypoints, features0.descriptors
        kps1, descs1_d = features1.keypoints, features1.descriptors
        scores_d, matches_d = kornia.feature.match_smnn(descs0_d, descs1_d)
        disk_kp0 = kps0[matches_d[:, 0], :]
        disk_kp1 = kps1[matches_d[:, 1], :]

        mixed_kp0 = torch.cat([origin_kp0, disk_kp0.cpu()], dim=0)
        mixed_kp1 = torch.cat([origin_kp1, disk_kp1.cpu()], dim=0)

        if len(mixed_kp0) < 300:
            if self.train_flag:
                torch.cuda.empty_cache()
                return self.__getitem__(np.random.randint(0, self.__len__() - 1, 1)[0])

        try:
            ransac = kornia.geometry.RANSAC(model_type='homography')
            _, mask = ransac(mixed_kp0, mixed_kp1)
            # H, mask = cv2.findHomography(mixed_kp0.numpy(), mixed_kp1.numpy(), cv2.RANSAC, 5.0)
            # mask = torch.tensor(mask, dtype=torch.bool)
            kp0 = mixed_kp0[mask.squeeze()]
            kp1 = mixed_kp1[mask.squeeze()]
            final_kp0, final_kp1 = self.remove_none_kps(kp0, kp1, image_0_origin.shape[-2], image_0_origin.shape[-1])

            if len(final_kp0) < 150:
                if self.train_flag:
                    torch.cuda.empty_cache()
                    return self.__getitem__(np.random.randint(0, self.__len__() - 1, 1)[0])
        except:
            final_kp0, final_kp1 = None, None

        # coarse_kp0 = overlap_kp[:, :2] * 8
        # coarse_kp1 = overlap_kp[:, 2:] * 8
        # fine_kp0 = grid_kp[:, :2]
        # fine_kp1 = grid_kp[:, 2:]

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

        T_0to1 = torch.tensor(self._compute_rel_pose(pose_0, pose_1), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        # lafs0, descs0 = kp_data_dict0.item()['lafs'], kp_data_dict0.item()['descriptor']
        # lafs1, descs1 = kp_data_dict1.item()['lafs'], kp_data_dict1.item()['descriptor']
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
            # 'lafs0': lafs0,
            # 'lafs1': lafs1,
            # 'descs0': descs0,
            # 'descs1': descs1,
            'origin_kp0': final_kp0,
            'origin_kp1': final_kp1,
        }
        return data


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    data_root = r'E:\Data\UnityCam\Stomach'
    # data_root = r'/home/zhangziang/Porject/DATA/UnityCam/train/Colon/'
    testclass = UnityDataset(data_root, mode='train', data_enhance=[20],
                             lighting_data=False, read_img_gray=False, train_flag=False)

    print(len(testclass))

    # disk = kornia.feature.DISK.from_pretrained('depth').to('cuda')

    # for idx, test in zip(numb, testclass):
    #     if idx > 1488:
    #         print(test['pair_names'])
    #         print(len(test['origin_kp0']))
    #         origin_kp0 = np.array(test['origin_kp0'].data.cpu())
    #         origin_kp1 = np.array(test['origin_kp1'].data.cpu())
    #
    #         image0 = np.array(test['image0'][0] * 255)
    #         image1 = np.array(test['image1'][0] * 255)
    #         img = make_matching_plot_fast_merge(image0, image1, origin_kp0, origin_kp1, color_set=[0, 0, 230])
    #
    #         # print(origin_kp0.shape[0])
    #         cv2.imshow('img', img)
    #         cv2.waitKey(1)

    for idx in range(len(testclass)):
        test = testclass[idx]
        print(test['pair_names'])
        print(len(test['origin_kp0']))
        try:
            origin_kp0 = np.array(test['origin_kp0'].data.cpu())
            origin_kp1 = np.array(test['origin_kp1'].data.cpu())

            image0 = np.array(test['image0'][0] * 255)
            image1 = np.array(test['image1'][0] * 255)
            img = make_matching_plot_fast_merge(image0, image1, origin_kp0, origin_kp1, color_set=[0, 0, 230])

            # print(origin_kp0.shape[0])
            cv2.imshow('img', img)
            cv2.waitKey(1)
        except:
            print('none gt')
