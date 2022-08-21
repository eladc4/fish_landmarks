
from datetime import datetime

import os
import numpy as np
import csv
import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets.transforms as np_transforms
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from scipy import linalg

from datasets.dataset_2d import Dataset2D
from datasets.dataset_2x2d import Dataset2x2D

from test import parse_args
from lib.config import cfg
from lib.config import update_config
import lib.models as models
from datasets.dataset_2d import show_image_with_keypoints


landmarks_legend = ['0: top lip',
                    '1: bottom lip',
                    '2: forward top fin',
                    '3: middle top fin',
                    '4: back top fin',
                    '5: top back fin',
                    '6: left fin - edge',
                    '7: right fin - edge',
                    '8: left fin - base',
                    '9: right fin - base']


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


def calc_3d_from_2x2d(landmarks1, landmarks2, cal_mat):
    P1 = cal_mat[:, 0].reshape((3, 4))
    P2 = cal_mat[:, 1].reshape((3, 4))
    out_3d = np.zeros((landmarks1.shape[0], 3))
    for landmark_idx in range(landmarks1.shape[0]):
        u1, v1 = landmarks1[landmark_idx, :]
        u2, v2 = landmarks2[landmark_idx, :]

        if np.isnan(u1) or np.isnan(u2):
            out_3d[landmark_idx, :] = np.nan
        else:
            out_3d[landmark_idx, :] = DLT(P1, P2, [u1, v1], [u2, v2])

    return out_3d


def calc_max_inds(heatmaps, heatmaps_weight):
    _output = np.zeros((heatmaps.shape[0], 2))
    for l in range(heatmaps.shape[0]):
        if heatmaps_weight[l] == 1:
            _output[l, 0] = 4 * np.argmax(np.amax(heatmaps[l, ...], axis=0))
            _output[l, 1] = 4 * np.argmax(np.amax(heatmaps[l, ...], axis=1))
        else:
            _output[l, 0] = np.nan
            _output[l, 1] = np.nan

    return _output


if __name__ == '__main__':
    if True:
        VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]
        is_train = False
        crop_ratio = 1.2 #2.4
        dataset = Dataset2x2D(train=False, val_split=VAL_SPLIT, reutrn_target_as_heatmaps=False,
                              num_input_images = cfg.MODEL.NUM_INPUT_IMAGES,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(127.5, 127.5)]),
                              np_transforms=[np_transforms.CropLandmarks(ratio=crop_ratio),
                                             np_transforms.Resize(image_size=(384, 384)),
                                             ],
                              )

        data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    else:
        # VAL_SPLIT = list(range(41))  # [0, 4, 12, 16, 21, 26, 33, 39]
        VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]
        dataset = Dataset2x2D(reutrn_target_as_heatmaps=False,
                              transform=transforms.Compose([transforms.ToTensor()]),
                              crop_landmarks=False, train=False, val_split=VAL_SPLIT,
                              crop_ratio=1.0)

        data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

    output_dir = "/results/dataset_eval"

    os.makedirs(output_dir, exist_ok=True)
    landmarks_dists_mat_dict = {}
    vid_name, image_index = 'None', -1
    with torch.no_grad():
        for i, (data, target, target_weight, cal_mats, pp) in enumerate(data_loader):
            target1 = target[0].cpu().detach().numpy()
            target2 = target[1].cpu().detach().numpy()
            target_weight = target_weight.cpu().detach().numpy()
            cal_mats = cal_mats.cpu().detach().numpy()

            for img_idx in range(target1.shape[0]):
                _vid_name = pp[0][0]['img_name'][img_idx].split(os.sep)[-3]
                if vid_name != _vid_name:
                    print(f'Analyzing {_vid_name}')
                image_index = image_index + 1 if vid_name == _vid_name else 0
                vid_name = _vid_name

                landmarks1 = target1[img_idx, ...]
                landmarks2 = target2[img_idx, ...]

                img1_size_y = pp[0][0]['img_size_y'][img_idx].numpy()
                img2_size_y = pp[0][1]['img_size_y'][img_idx].numpy()

                _landmarks1 = landmarks1.copy()
                _landmarks2 = landmarks2.copy()
                _landmarks1[:, 1] = img1_size_y - _landmarks1[:, 1]
                _landmarks2[:, 1] = img2_size_y - _landmarks2[:, 1]
                label_3d = calc_3d_from_2x2d(_landmarks1, _landmarks2, cal_mats[img_idx, ...])

                # update landmarks_dists_mat_dict
                _dist_mat = np.zeros((label_3d.shape[0], label_3d.shape[0]))
                for j in range(label_3d.shape[0]):
                    for k in range(label_3d.shape[0]):
                        _dist_mat[j, k] = np.linalg.norm(label_3d[j, ...] - label_3d[k, ...])

                if vid_name in landmarks_dists_mat_dict:
                    landmarks_dists_mat_dict[vid_name].append(_dist_mat)
                else:
                    landmarks_dists_mat_dict[vid_name] = [_dist_mat]

    # Analyze results
    for vid_name, landmarks_dists_mat in landmarks_dists_mat_dict.items():
        print(f'Exporting {vid_name}')
        landmarks_dists_mat = np.stack(landmarks_dists_mat)
        dists_mat = np.zeros(landmarks_dists_mat.shape[1:] + (3,))

        for j in range(landmarks_dists_mat.shape[1]):
            for k in range(landmarks_dists_mat.shape[2]):
                valid_dists = landmarks_dists_mat[~np.isnan(landmarks_dists_mat[:, j, k]), j, k]
                if valid_dists.shape[0] != 0:
                    dists_mat[j, k, 0] = valid_dists.mean()
                    dists_mat[j, k, 1] = valid_dists.std()
                    dists_mat[j, k, 2] = valid_dists.shape[0]

        header = ['MEAN'] + landmarks_legend + ['', 'STD'] + landmarks_legend + ['', '#POINTS'] + landmarks_legend
        with open(os.path.join(output_dir, f'{vid_name}_distances.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(10):
                writer.writerow([landmarks_legend[i]] + list(np.around(dists_mat[i, :, 0], decimals=2)) + [''] +
                                [landmarks_legend[i]] + list(np.around(dists_mat[i, :, 1], decimals=2)) + [''] +
                                [landmarks_legend[i]] + list(dists_mat[i, :, 2])
                                )

        os.makedirs(os.path.join(output_dir, vid_name), exist_ok=True)
        for i in range(10):
            for j in range(i+1, 10):
                valid_dists = landmarks_dists_mat[~np.isnan(landmarks_dists_mat[:, i, j]), i, j]
                plt.hist(valid_dists, bins=100, density=True)
                plt.title(f'distance between landmarks {landmarks_legend[i]} -> {landmarks_legend[j]}')
                plt.xlabel('[pixels]')
                plt.ylabel('prob')
                plt.grid()
                plt.savefig(os.path.join(output_dir, vid_name, f'dist_hist_{i}_{j}.png'))
                plt.close()

