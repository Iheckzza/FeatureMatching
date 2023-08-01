import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_parameters(Calibration_parameter_path):
    f = cv2.FileStorage(Calibration_parameter_path, cv2.FileStorage_READ)
    R = np.array(f.getNode('R').mat()).astype(float)
    T = np.array(f.getNode('T').mat()).astype(float)
    M1 = np.array(f.getNode('M1').mat()).astype(float)
    D1 = np.array(f.getNode('D1').mat()).astype(float)
    M2 = np.array(f.getNode('M2').mat()).astype(float)
    D2 = np.array(f.getNode('D2').mat()).astype(float)
    return R, T, M1, D1, M2, D2


def undistort(img, K, D, new_K=False):
    if new_K:
        h, w = img.shape[:2]
        n, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    else:
        Kn = K.copy()
    dst = cv2.undistort(img, K, D, None, Kn)
    if new_K:
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
    return dst, Kn


def main():
    original_dir = r'/home/zhangziang/Porject/DATA/origin_data/'
    save_dir = r'/home/zhangziang/Porject/DATA/origin_data/'

    for ii in sorted(os.listdir(save_dir)):
        data_name_list = os.path.join(save_dir, ii)
        for jj in sorted(os.listdir(data_name_list)):
            data_dir = os.path.join(data_name_list, jj) + r'/data'
            if os.path.exists(data_dir):
                origin_rgb_dir = data_dir + r'/rgb'
                cali_dir = os.path.join(data_name_list, jj) + r'/endoscope_calibration.yaml'
                save_undistort_dir = data_dir + r'/rgb_undistort'
                save_undistort_dir_l = save_undistort_dir + r'/left'
                save_undistort_dir_r = save_undistort_dir + r'/right'
                if not (os.path.exists(save_undistort_dir_l)):
                    os.mkdir(save_undistort_dir_l)
                if not (os.path.exists(save_undistort_dir_r)):
                    os.mkdir(save_undistort_dir_r)

                if int(ii[-1]) > 3 or (int(ii[-1]) == 3 and int(jj[-1]) >= 4):

                    print(origin_rgb_dir)
                    R, T, M1, D1, M2, D2 = get_parameters(cali_dir)
                    for kk in sorted(os.listdir(origin_rgb_dir)):
                        img = cv2.imread(os.path.join(origin_rgb_dir, kk), -1)
                        h, w = img.shape[0], img.shape[1]
                        img_l = img[:h // 2, :, :]
                        img_r = img[h // 2:, :, :]

                        img_l_undistort, _ = undistort(img_l, M1, D1)
                        img_r_undistort, _ = undistort(img_r, M2, D2)

                        img_l_undistort_resize = cv2.resize(img_l_undistort, (640, 480))
                        img_r_undistort_resize = cv2.resize(img_r_undistort, (640, 480))
                        cv2.imwrite(os.path.join(save_undistort_dir_l, kk), img_l_undistort_resize)
                        cv2.imwrite(os.path.join(save_undistort_dir_r, kk), img_r_undistort_resize)
                else:
                    continue





def temp():
    video_file = r'/home/zhangziang/Porject/DATA/origin_data/dataset_3/keyframe_4/data/rgb.mp4'
    save_file = r'/home/zhangziang/Porject/DATA/origin_data/dataset_3/keyframe_4/data/rgb/'
    cap = cv2.VideoCapture(video_file)

    i = 0
    while cap.isOpened():
        (flag, frame) = cap.read()  # 读取一张图像
        if (flag == True):
            fileName = ('Image{:0>6d}.png').format(i)
            save_path = os.path.join(save_file, fileName)
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            i += 1
        else:
            break


if __name__ == "__main__":
    main()
    # temp()
