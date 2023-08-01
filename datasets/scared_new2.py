from os import path as osp
from typing import Dict

import cv2
import numpy as np
import random
from unicodedata import name

import os
import torch
import torch.utils as utils
import json
import kornia

from scipy.spatial.transform import Rotation as R
from datasets.scared_toolkits.deeppruner_method.tiff_to_disp_sample import tiff_to_disp

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


def matches_split_list(root, images_lists, frame_data_lists, sence_images_lists, keypoints_lists, data_enhance, flag,
                       lighting_data,
                       scale):
    mulit_frames_data = []

    images_root = root + r'/raw_images'
    frames_root = root + r'/frame_data'
    sence_images_root = root + r'/sence_images'
    keypoints_root = root + r'/keypoints'

    images_lists.sort()
    frame_data_lists.sort()
    sence_images_lists.sort()
    keypoints_lists.sort()

    for ii, kk, ll, mm in zip(images_lists, frame_data_lists, sence_images_lists, keypoints_lists):
        ii = os.path.join(images_root, flag, ii)
        kk = os.path.join(frames_root, kk)
        ll = os.path.join(sence_images_root, flag, ll)
        mm = os.path.join(keypoints_root, mm)
        single_data = {'flag': flag, 'name': ii, 'frame_data': kk, 'sence_image': ll, 'kp_data': mm}
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


def read_images_gray(path, resize, augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def Loadframe(frame_dir, flag):
    with open(frame_dir, 'r') as load_f:
        load_dict = json.load(load_f)
        camera_pose = np.asarray(load_dict['camera-pose'])
        if flag == "Left":
            K = np.asarray(load_dict['camera-calibration']['KL'])
            D = np.asarray(load_dict['camera-calibration']['DL'])
        elif flag == 'Right':
            K = np.asarray(load_dict['camera-calibration']['KR'])
            D = np.asarray(load_dict['camera-calibration']['DR'])

    return K, D, camera_pose


def load_all_keyframe(keyframe_root, data_enhance, lighting_data, debug_flag):
    frames_root = keyframe_root + r'/frame_data'
    images_root = keyframe_root + r'/raw_images'
    sence_images_root = keyframe_root + r'/sence_images'
    kp_root = keyframe_root + r'/keypoints'

    left_images_lists = sorted(os.listdir(images_root + r'/Left'), key=lambda x: x.split()[-1])

    keypoints_lists = sorted(os.listdir(kp_root), key=lambda x: x.split()[-1])

    left_sence_images_root = sorted(os.listdir(sence_images_root + r'/Left'), key=lambda x: x.split()[-1])

    frame_data_lists = sorted(os.listdir(frames_root), key=lambda x: x.split()[-1])

    scale = np.log(len(left_images_lists)) * 0.08

    matches_final_left = matches_split_list(keyframe_root, left_images_lists, frame_data_lists,
                                            left_sence_images_root, keypoints_lists,
                                            data_enhance,
                                            'Left', lighting_data, scale)

    if debug_flag:
        matches_final_left = matches_final_left[:1]

    return matches_final_left


class ScaredDataset(utils.data.Dataset):
    def __init__(self, keyframe_root, mode=None, data_enhance=None, img_size=None, lighting_data=False,
                 with_sence=False, read_img_gray=True, debug_flag=False, train_flag=True, **kwargs):
        super().__init__()

        self.keyframe_root = keyframe_root

        self.mode = mode
        self.img_size = img_size
        self.read_img_gray = read_img_gray
        self.data_enhance = data_enhance
        self.lighting_data = lighting_data
        self.with_sence = with_sence

        self.debug_flag = debug_flag
        self.train_flag = train_flag
        self.matches_final = load_all_keyframe(self.keyframe_root, self.data_enhance, self.lighting_data,
                                               self.debug_flag)
        self.device = kornia.utils.helpers.get_cuda_device_if_available()

    def __len__(self):
        return len(self.matches_final)

    def _compute_rel_pose(self, pose0, pose1):
        return np.linalg.inv(np.matmul(np.linalg.inv(pose1), pose0))

    def convertK(self, K):
        K[0, :] /= 1280.0
        K[1, :] /= 1024.0
        K[0, :] *= 640
        K[1, :] *= 480
        return K

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

        ransac = kornia.geometry.RANSAC(model_type='homography')
        _, mask = ransac(src_pts, dst_pts)
        # H, mask = cv2.findHomography(np.array(src_pts.data.cpu()), np.array(dst_pts.data.cpu()), cv2.RANSAC, 5.0)
        # mask = torch.tensor(mask, dtype=torch.bool)

        origin_kp0 = src_pts[mask.squeeze()]
        origin_kp1 = dst_pts[mask.squeeze()]

        if self.train_flag:
            if len(origin_kp0) < 250:
                torch.cuda.empty_cache()
                return self.__getitem__(np.random.randint(0, self.__len__() - 1, 1)[0])

        if self.read_img_gray:
            image_0 = cv2.imread(image_0_dict['name'], cv2.IMREAD_GRAYSCALE)
            image_1 = cv2.imread(image_1_dict['name'], cv2.IMREAD_GRAYSCALE)
        else:
            image_0 = cv2.imread(image_0_dict['name'], -1)
            image_1 = cv2.imread(image_1_dict['name'], -1)

        if self.img_size is not None:
            image_0 = cv2.resize(image_0, self.img_size)
            image_1 = cv2.resize(image_1, self.img_size)

        image_0 = torch.from_numpy(image_0).float()[None] / 255 if image_0.shape[-1] != 3 else (
                torch.from_numpy(image_0).float() / 255).permute(2, 0, 1).contiguous()
        image_1 = torch.from_numpy(image_1).float()[None] / 255 if image_1.shape[-1] != 3 else (
                torch.from_numpy(image_1).float() / 255).permute(2, 0, 1).contiguous()

        frame_0_dir = image_0_dict['frame_data']
        frame_1_dir = image_1_dict['frame_data']

        K_0, D_0, T0 = Loadframe(frame_0_dir, image_0_dict['flag'])
        K_1, D_1, T1 = Loadframe(frame_1_dir, image_1_dict['flag'])

        K_0 = self.convertK(K_0)
        K_1 = self.convertK(K_1)

        t_0to1 = self._compute_rel_pose(T0, T1)
        t_1to0 = np.linalg.inv(t_0to1)
        T_0to1 = torch.tensor(t_0to1, dtype=torch.float32)
        T_1to0 = torch.tensor(t_1to0, dtype=torch.float32)

        sence_image_0 = np.load(image_0_dict['sence_image'], allow_pickle=True)
        sence_image_1 = np.load(image_1_dict['sence_image'], allow_pickle=True)
        sence_image_0 = torch.tensor(sence_image_0, dtype=torch.float32)
        sence_image_1 = torch.tensor(sence_image_1, dtype=torch.float32)
        data = {
            'image0': image_0,  # (1, h, w)
            'image1': image_1,
            'sence_image_0': sence_image_0,
            'sence_image_1': sence_image_1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'D0': D_0,
            'D1': D_1,
            'pair_id': idx,
            'pair_names': (image_0_dict['name'],
                           image_1_dict['name']),
            'origin_kp0': origin_kp0,
            'origin_kp1': origin_kp1,
        }
        return data


def warp2depth(grid_pt0_i, sence_img_0, T_0to1, K1, device):
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


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # data_root = r'/home/zhangziang/Porject/DATA/scared_modified/train/dataset_1/keyframe_1'
    data_root = r'E:\Data\processed\dataset_3\keyframe_2'
    testclass = ScaredDataset(data_root, mode='train', data_enhance=[10],
                              img_size=(640, 480), lighting_data=True, read_img_gray=True, debug_flag=False,
                              train_flag=False)

    test = testclass[15]
    image0 = np.array(test['image0'][0] * 255)
    image1 = np.array(test['image1'][0] * 255)
    # descs0, descs1 = test['descs0'], test['descs1']
    # lafs0, lafs1 = test['lafs0'], test['lafs1']
    # scores, matches = kornia.feature.match_mnn(descs0, descs1)
    #
    # # src_pts = lafs0[matches[:, 0].data.cpu(), :, 2]
    # # dst_pts = lafs1[matches[:, 1].data.cpu(), :, 2]
    #
    # src_pts = lafs0[matches[:, 0], :, 2]
    # dst_pts = lafs1[matches[:, 1], :, 2]
    #
    # ransac = kornia.geometry.RANSAC(model_type='homography', inl_th=2.0, batch_size=2048, max_iter=10,
    #                                 confidence=0.99, max_lo_iters=5)
    # _, mask = ransac(src_pts, dst_pts)
    #
    # origin_kp0 = np.array(src_pts[mask.squeeze()].data.cpu())
    # origin_kp1 = np.array(dst_pts[mask.squeeze()].data.cpu())
    origin_kp0 = np.array(test['origin_kp0'].data.cpu())
    origin_kp1 = np.array(test['origin_kp1'].data.cpu())

    img_origin = make_matching_plot_fast_merge(image0, image1, origin_kp0, origin_kp1, color_set=[0, 0, 230])
    print(origin_kp0.shape[0])

    cv2.imshow('img_origin', img_origin)
    cv2.waitKey(1)


