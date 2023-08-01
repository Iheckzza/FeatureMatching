import os
import numpy as np
import cv2
import random

import matplotlib.pyplot as plt
import torch
import kornia

from src.datasets.endoslam import EndoDataset
from src.datasets.unity_data import UnityDataset
from os import path as osp
from typing import Dict
from unicodedata import name
import torch.utils as utils
import json


def matches_split_list(root, images_lists, frame_data_lists, data_enhance, flag, lighting_data,
                       scale):
    mulit_frames_data = []

    images_root = root + r'/raw_images'
    frames_root = root + r'/frame_data'

    images_lists.sort()
    frame_data_lists.sort()

    for ii, kk in zip(images_lists, frame_data_lists):
        ii = os.path.join(images_root, flag, ii)
        kk = os.path.join(frames_root, kk)
        single_data = {'flag': flag, 'name': ii, 'frame_data': kk}
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

    left_images_lists = sorted(os.listdir(images_root + r'/Left'), key=lambda x: x.split()[-1])

    frame_data_lists = sorted(os.listdir(frames_root), key=lambda x: x.split()[-1])

    scale = len(data_enhance)

    matches_final_left = matches_split_list(keyframe_root, left_images_lists, frame_data_lists,
                                            data_enhance, 'Left', lighting_data, scale)
    return matches_final_left


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
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'pair_id': idx,
            'pair_names': (image_0_dict['name'],
                           image_1_dict['name'])
        }
        return data


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

    lafs_ny = np.array(lafs[0].data.cpu())
    out = np.array(descs[0].data.cpu())

    return {'lafs': lafs_ny, 'descriptor': out}


def test_show():
    original_dir = r'E:\Data\processed'
    # original_dir = r'/home/zhangziang/Porject/DATA/scared_modified/train/'

    device = kornia.utils.helpers.get_cuda_device_if_available()

    enhance_list = [10, 20]

    for ii in sorted(os.listdir(original_dir)):
        data_name_list = os.path.join(original_dir, ii)
        for jj in sorted(os.listdir(data_name_list)):
            keyframe_list = os.path.join(data_name_list, jj)

            for enhance in enhance_list:
                scared = ScaredDataset(keyframe_list, mode='train', data_enhance=[enhance],
                                       data_type='single',
                                       img_size=(640, 480), lighting_data=False)

                save_matches_dir = keyframe_list + r'/matches'
                keypoints_dir = keyframe_list + r'/keypoints'

                for idx, current_pair in enumerate(scared):
                    print(current_pair['pair_names'])

                    kp0_dir = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    kp1_dir = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx + 1, '.npy'))
                    load_descs0 = np.load(kp0_dir, allow_pickle=True)
                    load_descs1 = np.load(kp1_dir, allow_pickle=True)

                    lafs0, descs0 = load_descs0.item()['lafs'], load_descs0.item()['descriptor']
                    lafs1, descs1 = load_descs1.item()['lafs'], load_descs1.item()['descriptor']

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
                    print(len(kpts0))

                    # if idx == len(scared) - 1:
                    #     descs = cal_kps(current_pair['image0'], device)
                    #     descs1 = cal_kps(current_pair['image1'], device)
                    #     save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    #     save_kp_path_np1 = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx + 1, '.npy'))
                    #     np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)
                    #     np.save(save_kp_path_np1, descs1, allow_pickle=True, fix_imports=True)
                    # else:
                    #     descs = cal_kps(current_pair['image0'], device)
                    #     save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    #     np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)

                    kpts0c, kpts1c, H = cal_matches_kps(current_pair['image0'], current_pair['image1'], device)

                    kpts0 = (kpts0 // 8) * 8
                    kpts1 = (kpts1 // 8) * 8
                    img0 = np.array(current_pair['image0'][0] * 255)
                    img1 = np.array(current_pair['image1'][0] * 255)
                    img = make_matching_plot_fast_merge(img0, img1, kpts0, kpts1)
                    img_c = make_matching_plot_fast_merge(img0, img1, kpts0c, kpts1c)
                    cv2.imshow('img', img)
                    cv2.imshow('imgc', img_c)
                    cv2.waitKey(0)
                break
            break
        break


def create_kps():
    original_dir = r'E:\Data\processed'
    # original_dir = r'/home/zhangziang/Porject/DATA/scared_modified/train/'

    device = 'cuda:0'

    for ii in sorted(os.listdir(original_dir)):
        data_name_list = os.path.join(original_dir, ii)
        for jj in sorted(os.listdir(data_name_list)):
            keyframe_list = os.path.join(data_name_list, jj)

            scared = ScaredDataset(keyframe_list, mode='train', data_enhance=[1],
                                   data_type='single',
                                   img_size=(640, 480), lighting_data=False)

            keypoints_dir = keyframe_list + r'/keypoints'
            if not os.path.exists(keypoints_dir):
                os.mkdir(keypoints_dir)

            for idx, current_pair in enumerate(scared):
                print(current_pair['pair_names'])

                if idx == len(scared) - 1:
                    descs = cal_kps(current_pair['image0'], device)
                    descs1 = cal_kps(current_pair['image1'], device)
                    save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    save_kp_path_np1 = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx + 1, '.npy'))
                    np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)
                    np.save(save_kp_path_np1, descs1, allow_pickle=True, fix_imports=True)
                else:
                    descs = cal_kps(current_pair['image0'], device)
                    save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)


def create_endokps():
    # original_dir = r'E:\Data\EndoSLAM\Cameras\HighCam'
    original_dir = r'E:\Data\EndoSLAM\Cameras\LowCam'
    device = 'cuda:0'

    for ii in sorted(os.listdir(original_dir)):
        data_name_list = os.path.join(original_dir, ii)
        for jj in sorted(os.listdir(data_name_list)):
            keyframe_list = os.path.join(data_name_list, jj)
            keypoints_dir = keyframe_list + r'/keypoints'
            if not os.path.exists(keypoints_dir):
                os.mkdir(keypoints_dir)
            endo = EndoDataset(keyframe_list, mode='train', data_enhance=[1], lighting_data=False, read_img_gray=True)

            for idx, current_pair in enumerate(endo):
                print(current_pair['pair_names'])

                img_0_name_id = current_pair['pair_names'][0][-10:-4]
                img_1_name_id = current_pair['pair_names'][1][-10:-4]

                if idx == len(endo) - 1:
                    descs = cal_kps(current_pair['image0'], device)
                    descs1 = cal_kps(current_pair['image1'], device)
                    save_kp_path_np = os.path.join(keypoints_dir, 'keypoints_' + str(img_0_name_id) + '.npy')
                    save_kp_path_np1 = os.path.join(keypoints_dir, 'keypoints_' + str(img_1_name_id) + '.npy')

                    np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)
                    np.save(save_kp_path_np1, descs1, allow_pickle=True, fix_imports=True)
                else:
                    descs = cal_kps(current_pair['image0'], device)
                    save_kp_path_np = os.path.join(keypoints_dir, 'keypoints_' + str(img_0_name_id) + '.npy')
                    np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)


def create_unitykps():
    # original_dir = r'E:\Data\EndoSLAM\Cameras\HighCam'
    original_dir = r'E:\Data\UnityCam'
    device = 'cuda:0'

    for ii in sorted(os.listdir(original_dir)):
        keyframe_list = os.path.join(original_dir, ii)
        keypoints_dir = keyframe_list + r'/keypoints'
        frame_dir = keyframe_list + r'/Frames'

        if not os.path.exists(keypoints_dir):
            os.mkdir(keypoints_dir)

        for idx, current_ in enumerate(os.listdir(frame_dir)):
            current_frame = os.path.join(frame_dir,current_)
            print(current_frame)

            img_name_id = current_frame.split('\\')[-1][:-4].split('_')[-1]

            img = cv2.imread(current_frame, 0)
            img_t = torch.from_numpy(img).float()[None].to(device)

            descs = cal_kps(img_t, device)
            save_kp_path_np = os.path.join(keypoints_dir, 'keypoints_' + str(img_name_id) + '.npy')
            np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)

def main():
    original_dir = r'E:\Data\processed'
    # original_dir = r'/home/zhangziang/Porject/DATA/scared_modified/train/'

    device = kornia.utils.helpers.get_cuda_device_if_available()

    enhance_list = [1, 5, 10, 20]

    for ii in sorted(os.listdir(original_dir)):
        data_name_list = os.path.join(original_dir, ii)
        for jj in sorted(os.listdir(data_name_list)):
            keyframe_list = os.path.join(data_name_list, jj)

            for enhance in enhance_list:
                scared = ScaredDataset(keyframe_list, mode='train', data_enhance=[enhance],
                                       data_type='single',
                                       img_size=(640, 480), lighting_data=False)

                save_matches_dir = keyframe_list + r'/matches'
                keypoints_dir = keyframe_list + r'/keypoints'
                save_n_path = save_matches_dir + f'/matches_n={enhance}'
                if not os.path.exists(save_n_path):
                    os.mkdir(save_n_path)
                if not os.path.exists(save_matches_dir):
                    os.mkdir(save_matches_dir)
                if not os.path.exists(keypoints_dir):
                    os.mkdir(keypoints_dir)

                for idx, current_pair in enumerate(scared):
                    print(current_pair['pair_names'])

                    kp0_dir = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    kp1_dir = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx + 3, '.npy'))
                    load_descs0 = np.load(kp0_dir, allow_pickle=True)
                    load_descs1 = np.load(kp1_dir, allow_pickle=True)

                    lafs0, descs0 = load_descs0.item()['lafs'], load_descs0.item()['descriptor']
                    lafs1, descs1 = load_descs1.item()['lafs'], load_descs1.item()['descriptor']

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
                    print(len(kpts0))

                    # if idx == len(scared) - 1:
                    #     descs = cal_kps(current_pair['image0'], device)
                    #     descs1 = cal_kps(current_pair['image1'], device)
                    #     save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    #     save_kp_path_np1 = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx + 1, '.npy'))
                    #     np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)
                    #     np.save(save_kp_path_np1, descs1, allow_pickle=True, fix_imports=True)
                    # else:
                    #     descs = cal_kps(current_pair['image0'], device)
                    #     save_kp_path_np = os.path.join(keypoints_dir, "keypoints_{:06d}{}".format(idx, '.npy'))
                    #     np.save(save_kp_path_np, descs, allow_pickle=True, fix_imports=True)

                    # kpts0, kpts1, H = cal_matches_kps(current_pair['image0'], current_pair['image1'], device)

                    img0 = np.array(current_pair['image0'][0] * 255)
                    img1 = np.array(current_pair['image1'][0] * 255)
                    img = make_matching_plot_fast_merge(img0, img1, kpts0, kpts1)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
                break
            break
        break


if __name__ == "__main__":
    # main()
    # create_kps()
    # create_endokps()
    create_unitykps()
    # test_show()
