from os import path as osp
from typing import Dict

import cv2
import numpy as np
from unicodedata import name

import os
import torch
import torch.utils as utils
import json
import kornia


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
        for i in range(len(mulit_frames_data) - 1):

            if lighting_data:
                k0 = i * ii
                k1 = ii * (i + 1)
            else:
                k0 = i
                k1 = i + int(ii)
            if k1 < len(mulit_frames_data) and k0 % scale == 0:
                matches_final.append({'img0': mulit_frames_data[k0], 'img1': mulit_frames_data[k1]})

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
        elif flag == 'Right':
            K = np.asarray(load_dict['camera-calibration']['KR'])

    return K, camera_pose


def load_all_keyframe(keyframe_root, data_enhance, lighting_data):
    frames_root = keyframe_root + r'/frame_data'
    images_root = keyframe_root + r'/raw_images'
    sence_images_root = keyframe_root + r'/sence_images'
    kp_root = keyframe_root + r'/keypoints'

    left_images_lists = sorted(os.listdir(images_root + r'/Left'), key=lambda x: x.split()[-1])

    keypoints_lists = sorted(os.listdir(kp_root), key=lambda x: x.split()[-1])

    left_sence_images_root = sorted(os.listdir(sence_images_root + r'/Left'), key=lambda x: x.split()[-1])

    frame_data_lists = sorted(os.listdir(frames_root), key=lambda x: x.split()[-1])

    # scale = int(np.log(len(left_images_lists))) + len(data_enhance)
    scale = len(data_enhance)

    matches_final_left = matches_split_list(keyframe_root, left_images_lists, frame_data_lists,
                                            left_sence_images_root, keypoints_lists,
                                            data_enhance,
                                            'Left', lighting_data, scale)

    return matches_final_left


def cal_kps(kp_data_dict0, kp_data_dict1, device):
    lafs0, descs0 = kp_data_dict0.item()['lafs'], kp_data_dict0.item()['descriptor']
    lafs1, descs1 = kp_data_dict1.item()['lafs'], kp_data_dict1.item()['descriptor']

    descs0_t = torch.tensor(descs0, dtype=torch.float32, device=device)
    descs1_t = torch.tensor(descs1, dtype=torch.float32, device=device)

    with torch.no_grad():
        scores, matches = kornia.feature.match_mnn(descs0_t, descs1_t)

    src_pts = lafs0[np.array(matches[:, 0].data.cpu()), :, 2]
    dst_pts = lafs1[np.array(matches[:, 1].data.cpu()), :, 2]

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = np.array(mask, dtype=bool)

    kpts0 = src_pts[mask.squeeze()]
    kpts1 = dst_pts[mask.squeeze()]

    return kpts0, kpts1


def remove_overlap_kps(origin_kp0, origin_kp1):
    overlap_kp0 = origin_kp0 // 8
    overlap_kp1 = origin_kp1 // 8
    overlapkp_total = np.concatenate([overlap_kp0, overlap_kp1], axis=1)
    orikp_total = np.concatenate([origin_kp0, origin_kp1], axis=1)
    _, idx0 = np.unique(overlapkp_total[:, :2], axis=0, return_index=True)
    overlapkp_cache = overlapkp_total[idx0]
    orikp_cache = orikp_total[idx0]
    _, idx1 = np.unique(overlapkp_cache[:, 2:], axis=0, return_index=True)
    overlapkp_final = overlapkp_cache[idx1]
    orikp_final = orikp_cache[idx1]

    return overlapkp_final, orikp_final


class ScaredDataset(utils.data.Dataset):
    def __init__(self, keyframe_root, mode=None, data_enhance=None, data_type=None, img_size=None, lighting_data=False,
                 with_sence=False, read_img_gray=True, **kwargs):
        super().__init__()

        self.keyframe_root = keyframe_root

        self.mode = mode
        self.img_size = img_size
        self.read_img_gray = read_img_gray
        self.data_enhance = data_enhance
        self.data_type = data_type
        self.lighting_data = lighting_data
        self.with_sence = with_sence

        self.matches_final = load_all_keyframe(self.keyframe_root, self.data_enhance, self.lighting_data)
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

        if self.read_img_gray:
            image_0 = cv2.imread(image_0_dict['name'], cv2.IMREAD_GRAYSCALE)
            image_1 = cv2.imread(image_1_dict['name'], cv2.IMREAD_GRAYSCALE)
            if self.img_size is not None:
                image_0 = cv2.resize(image_0, self.img_size)
                image_1 = cv2.resize(image_1, self.img_size)

            image_0 = torch.from_numpy(image_0).float()[None] / 255
            image_1 = torch.from_numpy(image_1).float()[None] / 255
        else:
            image_0 = cv2.imread(image_0_dict['name'], -1)
            image_1 = cv2.imread(image_1_dict['name'], -1)

            image_0 = (torch.from_numpy(image_0).float() / 255).permute(2, 0, 1).contiguous()
            image_1 = (torch.from_numpy(image_1).float() / 255).permute(2, 0, 1).contiguous()

        sence_image_0 = np.load(image_0_dict['sence_image'], allow_pickle=True)
        sence_image_1 = np.load(image_1_dict['sence_image'], allow_pickle=True)

        sence_image_0 = torch.tensor(sence_image_0, dtype=torch.float32)
        sence_image_1 = torch.tensor(sence_image_1, dtype=torch.float32)

        frame_0_dir = image_0_dict['frame_data']
        frame_1_dir = image_1_dict['frame_data']

        K_0, T0 = Loadframe(frame_0_dir, image_0_dict['flag'])
        K_1, T1 = Loadframe(frame_1_dir, image_1_dict['flag'])

        K_0 = self.convertK(K_0)
        K_1 = self.convertK(K_1)

        T_0to1 = torch.tensor(self._compute_rel_pose(T0, T1), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        kp_data_dict0 = np.load(image_0_dict['kp_data'], allow_pickle=True)
        kp_data_dict1 = np.load(image_1_dict['kp_data'], allow_pickle=True)
        origin_kp0, origin_kp1 = cal_kps(kp_data_dict0, kp_data_dict1, self.device)

        overlap_kp, grid_kp = remove_overlap_kps(origin_kp0, origin_kp1)
        coarse_kp0 = overlap_kp[:, :2] * 8
        coarse_kp1 = overlap_kp[:, 2:] * 8
        fine_kp0 = grid_kp[:, :2]
        fine_kp1 = grid_kp[:, 2:]

        lists_f0 = np.array(overlap_kp[:, :2][:, 0] + 80 * overlap_kp[:, :2][:, 1], dtype=np.int32)
        lists_f1 = np.array(overlap_kp[:, 2:][:, 0] + 80 * overlap_kp[:, 2:][:, 1], dtype=np.int32)
        fine_mtx_0 = np.zeros((60 * 80, 2), dtype=np.float32)
        fine_mtx_1 = np.zeros((60 * 80, 2), dtype=np.float32)

        fine_mtx_0[lists_f0] = fine_kp0
        fine_mtx_1[lists_f1] = fine_kp1

        data = {
            'image0': image_0,  # (1, h, w)
            'image1': image_1,
            'sence_image_0': sence_image_0,
            'sence_image_1': sence_image_1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'pair_id': idx,
            'pair_names': (image_0_dict['name'],
                           image_1_dict['name']),
            'origin_kp0': origin_kp0,
            'origin_kp1': origin_kp1,
            'coarse_kp0': coarse_kp0,
            'coarse_kp1': coarse_kp1,
            'fine_kp0': fine_kp0,
            'fine_kp1': fine_kp1,
            'fine_mtx_0': fine_mtx_0,
            'fine_mtx_1': fine_mtx_1,
            'lists_f0': lists_f0,
            'lists_f1': lists_f1
        }
        return data


if __name__ == "__main__":
    # data_root = r'/home/zhangziang/Porject/DATA/scared_modified/train/'
    data_root = r'E:\Data\processed\dataset_1\keyframe_1'
    testclass = ScaredDataset(data_root, mode='train', data_enhance=[1, 5, 15, 20, 30],
                              img_size=(640, 480), lighting_data=True, read_img_gray=False)

    print(len(testclass))

    test = testclass[10]
    #
    # image0 = np.array(test['image0'][0] * 255)
    # image1 = np.array(test['image1'][0] * 255)
    #
    # kpts0 = test['fine_kp0']
    # kpts1 = test['fine_kp1']
    #
    # img = make_matching_plot_fast_merge(image0, image1, kpts0, kpts1, color_set=[0, 0, 230])
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    #
    # print(len(kpts0))
