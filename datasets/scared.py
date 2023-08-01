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


def matches_split_list(root, images_lists, frame_data_lists, sence_images_lists, data_enhance, flag, lighting_data,
                       scale):
    mulit_frames_data = []

    images_root = root + r'/raw_images'
    frames_root = root + r'/frame_data'
    sence_images_root = root + r'/sence_images'

    images_lists.sort()
    frame_data_lists.sort()
    sence_images_lists.sort()

    for ii, kk, ll in zip(images_lists, frame_data_lists, sence_images_lists):
        ii = os.path.join(images_root, flag, ii)
        kk = os.path.join(frames_root, kk)
        ll = os.path.join(sence_images_root, flag, ll)
        single_data = {'flag': flag, 'name': ii, 'frame_data': kk, 'sence_image': ll}
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


def load_all_keyframe(keyframe_root, data_enhance, data_type, lighting_data):
    frames_root = keyframe_root + r'/frame_data'
    images_root = keyframe_root + r'/raw_images'
    sence_images_root = keyframe_root + r'/sence_images'

    left_images_lists = sorted(os.listdir(images_root + r'/Left'), key=lambda x: x.split()[-1])
    right_images_lists = sorted(os.listdir(images_root + r'/Right'), key=lambda x: x.split()[-1])

    left_sence_images_root = sorted(os.listdir(sence_images_root + r'/Left'), key=lambda x: x.split()[-1])
    right_sence_images_root = sorted(os.listdir(sence_images_root + r'/Right'), key=lambda x: x.split()[-1])

    frame_data_lists = sorted(os.listdir(frames_root), key=lambda x: x.split()[-1])

    # scale = int(np.log(len(left_images_lists))) + len(data_enhance)
    scale = len(data_enhance)

    matches_final_left = matches_split_list(keyframe_root, left_images_lists, frame_data_lists,
                                            left_sence_images_root,
                                            data_enhance,
                                            'Left', lighting_data, scale)
    matches_final_right = matches_split_list(keyframe_root, right_images_lists, frame_data_lists,
                                             right_sence_images_root,
                                             data_enhance,
                                             'Right', lighting_data, scale)

    matches_final = []
    if data_type == 'single':
        matches_final = matches_final_left
    elif data_type == 'double':
        matches_final = matches_final_left + matches_final_right

    return matches_final


class ScaredDataset(utils.data.Dataset):
    def __init__(self, keyframe_root, mode=None, data_enhance=None, data_type=None, img_size=None, lighting_data=False,
                 **kwargs):
        super().__init__()

        self.keyframe_root = keyframe_root

        self.mode = mode
        self.img_size = img_size
        self.data_enhance = data_enhance
        self.data_type = data_type
        self.lighting_data = lighting_data

        self.matches_final = load_all_keyframe(self.keyframe_root, self.data_enhance, self.data_type,
                                               self.lighting_data)

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
        image_0 = cv2.imread(image_0_dict['name'], cv2.IMREAD_GRAYSCALE)
        image_1 = cv2.imread(image_1_dict['name'], cv2.IMREAD_GRAYSCALE)

        sence_image_0 = np.load(image_0_dict['sence_image'], allow_pickle=True)
        sence_image_1 = np.load(image_1_dict['sence_image'], allow_pickle=True)

        sence_image_0 = torch.tensor(sence_image_0, dtype=torch.float32)
        sence_image_1 = torch.tensor(sence_image_1, dtype=torch.float32)

        if self.img_size is not None:
            image_0 = cv2.resize(image_0, self.img_size)
            image_1 = cv2.resize(image_1, self.img_size)

        image_0 = torch.from_numpy(image_0).float()[None] / 255
        image_1 = torch.from_numpy(image_1).float()[None] / 255

        frame_0_dir = image_0_dict['frame_data']
        frame_1_dir = image_1_dict['frame_data']

        K_0, T0 = Loadframe(frame_0_dir, image_0_dict['flag'])
        K_1, T1 = Loadframe(frame_1_dir, image_1_dict['flag'])

        K_0 = self.convertK(K_0)
        K_1 = self.convertK(K_1)

        T_0to1 = torch.tensor(self._compute_rel_pose(T0, T1), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

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
                           image_1_dict['name'])
        }
        return data


if __name__ == "__main__":
    # data_root = r'/home/zhangziang/Porject/Loftr-modified/original_data/scared_modified/train/dataset_1/keyframe_3'
    data_root = r'E:\Data\processed\dataset_2\keyframe_2'
    testclass = ScaredDataset(data_root, mode='train', data_enhance=[1],
                              data_type='single',
                              img_size=(640, 480), lighting_data=False)

    test = testclass[0]
    # sence_image_0 = np.array(test["sence_image_0"])
    # print(sence_image_0.shape)
    # cv2.imshow("sence_image_0", sence_image_0[2])
    #
    # sence_image_1 = np.array(test["sence_image_1"])
    # print(sence_image_1.shape)
    # cv2.imshow("sence_image_1", sence_image_1[2])
    # cv2.waitKey(0)

    # for test in testclass:
    #     print(test['pair_names'])

    print(len(testclass))
