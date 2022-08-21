import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import csv
from scipy import linalg

from videos_and_csv_labels import get_videos_and_csv_labels_list
from gen_2d_dataset import show_image_with_keypoints


def calc_epipolar_points(u1, F, u2):
    x1 = np.ones((3, 1))
    x1[:2, 0] = u1
    a, b, c = list((x1.transpose() @ F).reshape(-1))
    v2 = [-(a * u2[0] + c) / b, -(a * u2[1] + c) / b]
    return u2, v2


def calc_fundamental_matrix(landmarks0, landmarks1):
    # estimate the fundamental matrix:
    A = []
    for (u1, v1), (u2, v2) in zip(landmarks0, landmarks1):
        if not np.isnan(u1):
            A.append(np.array([u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, 1]))
    A = np.stack(A)
    U, s, Vt = np.linalg.svd(A.transpose() @ A)
    F = Vt[-1, :].reshape((3, 3))
    return F


def read_cal_matrix(cal_file):
    with open(cal_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        cal_mat = np.ones((12, 2))
        for i, row in enumerate(csv_reader):
            cal_mat[i, :] = np.array(row)
    return cal_mat


def read_labels(label_file, vid_height):
    num_keypoints = 17
    # relevant_keypoints = np.arange(num_keypoints)
    relevant_keypoints = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    with open(label_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        cam1_keypoints, cam2_keypoints = [], []
        for row in csv_reader:
            if len(row) > 4*num_keypoints:
                raise Exception('More than 17 keypoints')
            # need to flip y-axis, as axes in matlab are reversed compared to python
            if vid_height is None:
                cam1_keypoints.append(np.array([(float(row.get(f'pt{i}_cam1_X', float('nan'))),
                                                 float(row.get(f'pt{i}_cam1_Y', float('nan'))))
                                                for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
                cam2_keypoints.append(np.array([(float(row.get(f'pt{i}_cam2_X', float('nan'))),
                                                 float(row.get(f'pt{i}_cam2_Y', float('nan'))))
                                                for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
            else:
                cam1_keypoints.append(np.array([(float(row.get(f'pt{i}_cam1_X', float('nan'))),
                                                 vid_height-float(row.get(f'pt{i}_cam1_Y', float('nan'))))
                                                for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
                cam2_keypoints.append(np.array([(float(row.get(f'pt{i}_cam2_X', float('nan'))),
                                                 vid_height-float(row.get(f'pt{i}_cam2_Y', float('nan'))))
                                                for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])

    return cam1_keypoints, cam2_keypoints


def read_labels_3d(xyz_file, vid_height):
    num_keypoints = 17
    # relevant_keypoints = np.arange(num_keypoints)
    relevant_keypoints = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    with open(xyz_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        xyz_keypoints = []
        for row in csv_reader:
            if len(row) > 3*num_keypoints:
                raise Exception('More than 17 keypoints')
            # need to flip y-axis, as axes in matlab are reversed compared to python
            if vid_height is None:
                xyz_keypoints.append(np.array([(float(row.get(f'pt{i}_X', float('nan'))),
                                                float(row.get(f'pt{i}_Y', float('nan'))),
                                                float(row.get(f'pt{i}_Z', float('nan'))))
                                               for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
            else:
                xyz_keypoints.append(np.array([(float(row.get(f'pt{i}_X', float('nan'))),
                                                vid_height-float(row.get(f'pt{i}_Y', float('nan'))),
                                                float(row.get(f'pt{i}_Z', float('nan'))))
                                               for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])

    return xyz_keypoints


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]

########################################
# # pt1_cam1_X	pt1_cam1_Y	pt1_cam2_X	pt1_cam2_Y
# u1 = 669.77498
# v1 = 161.206561
# u2 = 729.445158
# v2 = 135.98186
#
# # pt1_X	pt1_Y	pt1_Z
# x = 8.538395
# y = 0.242166
# z = 5.665664
#
# PP = np.array([[32.029, 22.396],
#                [2.7463, -0.99966],
#                [11.625, -23.419],
#                [325.58, 598.14],
#                [0.74847, 0.55813],
#                [33.301, 30.967],
#                [-5.6794, -2.978],
#                [178.21, 126.66],
#                [0.0064032, -0.0053584],
#                [0.0004211, 0.0010773],
#                [-0.010779, -0.0096336],
#                [1,1]])
#
# P1 = PP[:, 0].reshape((3, 4))
# P2 = PP[:, 1].reshape((3, 4))
#
# p = DLT(P1, P2, (u1, v1), (u2, v2))
#
# print(p)
# print((x, y, z))
##################################

vid_and_label = get_videos_and_csv_labels_list(base_path=r"C:\eladc\project\dataset_src\all_dig_data")[0]
cams_vidcap = [cv2.VideoCapture(vid_and_label.video1),
               cv2.VideoCapture(vid_and_label.video2)]
vid_height = cams_vidcap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
cams_kpts = read_labels(vid_and_label.label_csv, None)
cal_mat = read_cal_matrix(vid_and_label.cal_csv)
xyz_points = read_labels_3d(vid_and_label.label_csv.replace('xypts', 'xyzpts'), None)

imgs = []
for cam_index, cam_vidcap in enumerate(cams_vidcap):
    imgs.append([])
    while True:
        success, image = cam_vidcap.read()
        if not success:
            break
        imgs[-1].append(Image.fromarray(image))

img_idx = 0
landmark_idx = 0
u1, v1 = cams_kpts[0][img_idx][landmark_idx, :]
u2, v2 = cams_kpts[1][img_idx][landmark_idx, :]
P1 = cal_mat[:, 0].reshape((3, 4))
P2 = cal_mat[:, 1].reshape((3, 4))

x = DLT(P1, P2, [u1, v1], [u2, v2])

A = np.stack([v1*P1[2, :] - P1[1, :],
              P1[0, :] - u1*P1[2, :],
              v2 * P2[2, :] - P2[1, :],
              P2[0, :] - u2 * P2[2, :]
              ])

U, s, Vt = np.linalg.svd(A)
xn = Vt[-1, :]
_x = xn[:-1] / xn[-1]

print(x)
print(_x)
print(xyz_points[img_idx][landmark_idx, :])

# estimate the fundamental matrix:
F = calc_fundamental_matrix(cams_kpts[0][img_idx], cams_kpts[1][img_idx])
u2, v2 = calc_epipolar_points(cams_kpts[0][img_idx][0, :], F, [0, 1280])

fixed_labels = cams_kpts[1][img_idx].copy()
fixed_labels[:, 1] = vid_height - fixed_labels[:, 1]
img = show_image_with_keypoints(imgs[1][img_idx], fixed_labels)
draw = ImageDraw.Draw(img)
draw.line((u2[0], vid_height-v2[0], u2[1], vid_height-v2[1]), fill=128)
img.show()

# cam_idx = 1
# show_image_with_keypoints(imgs[cam_idx][img_idx], cams_kpts[cam_idx][img_idx]).show()
a=1
