
import os
from glob import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import datasets.transforms as np_preprocess
from PIL import Image, ImageOps
import cv2 as cv

from scripts.gen_2d_dataset import show_image_with_keypoints


class Dataset2x2D(Dataset):
    # Landmarks:
    # 0: top lip
    # 1: bottom lip
    # 2: forward top fin
    # 3: middle top fin
    # 4: back top fin
    # 5: top back fin
    # 6: left fin - edge
    # 7: right fin - edge
    # 8: left fin - base
    # 9: right fin - base
    _LANDMARK_FLIP = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8]

    def __init__(self,
                 imgs_folder="/data/projects/swat/users/eladc/project/dataset_2x2d",
                 label_file="/data/projects/swat/users/eladc/project/dataset_2x2d/labels.pickle",
                 cal_file="/data/projects/swat/users/eladc/project/dataset_2x2d/cal_mats.pickle",
                 gape_file="/data/projects/swat/users/eladc/project/dataset_2x2d/gape.pickle",
                 plate_file="/data/projects/swat/users/eladc/project/dataset_2x2d/plate.pickle",
                 transform=None, np_transforms=None, reutrn_target_as_heatmaps=True, heatmap_size=(96, 96),
                 sigma=1, image_size=384, train=True, val_split=0.2, num_input_images=3,
                 use_prev_hm_input=False, fixed_crop_per_vid=False, fixed_crop_ratio=2.0):

        self.imgs_folder = imgs_folder

        vids = []
        num_images = 0
        for _vid_path in glob(os.path.join(imgs_folder, "*")):
            if os.path.isdir(_vid_path):
                vids.append([])
                for _cam_path in glob(os.path.join(_vid_path, "*")):
                    if os.path.isdir(_vid_path):
                        vids[-1].append([])
                        for _img in glob(os.path.join(_cam_path, "*")):
                            vids[-1][-1].append(_img)
                            num_images += 1
        # imgs = glob(os.path.join(imgs_folder, "**", "*.jpg"))
        self.cal_file = cal_file
        with open(cal_file, 'rb') as f:
            cal_mats = pickle.load(f)
        self.label_file = label_file
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
        with open(gape_file, 'rb') as f:
            gape = pickle.load(f)
        with open(plate_file, 'rb') as f:
            plate = pickle.load(f)
        self.transform = transform
        self.np_transforms = np_transforms
        self.reutrn_target_as_heatmaps = reutrn_target_as_heatmaps
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.image_size = image_size
        self.train = train
        self.use_different_joints_weight = False
        self.joints_weight = None
        self.num_input_images = num_input_images
        self.fixed_crop_per_vid = fixed_crop_per_vid
        self.fixed_crop_ratio = fixed_crop_ratio
        assert not (use_prev_hm_input and num_input_images == 1)
        self.use_prev_hm_input = use_prev_hm_input

        # verify labels and data are the same size
        if len(labels) != len(vids) or any([len(c1) != len(c2) for c1, c2 in zip(labels, vids)]) or \
                any([len(i1) != len(i2) for c1, c2 in zip(labels, vids) for i1, i2 in zip(c1, c2)]):
            raise Exception('Mismatch between images and labels')

        # remove labels and corresponding imgs
        for vid_index, vid_labels in enumerate(labels):
            remove_inds = set()
            for cam_index, cam_labels in enumerate(vid_labels):
                for img_index, _label in enumerate(cam_labels):
                    if np.all(np.isnan(_label)):
                        remove_inds.add(img_index)
            if remove_inds:
                for cam_index, cam_labels in enumerate(vid_labels):
                    _imgs, _labels, _gape_list, _plate_list = [], [], [], []
                    for img_index, (_label, _img, _gape, _plate) in enumerate(zip(cam_labels,
                                                                                  vids[vid_index][cam_index],
                                                                                  gape[vid_index][cam_index],
                                                                                  plate[vid_index][cam_index])):
                        if img_index not in remove_inds:
                            _imgs.append(_img)
                            _labels.append(_label)
                            _gape_list.append(_gape)
                            _plate_list.append(_plate)
                        else:
                            num_images = num_images - 1
                    vids[vid_index][cam_index] = _imgs
                    labels[vid_index][cam_index] = _labels
                    gape[vid_index][cam_index] = _gape_list
                    plate[vid_index][cam_index] = _plate_list

        if val_split is not None:
            # select inds for validation and training
            if isinstance(val_split, (tuple, list)):
                val_inds = val_split
            else:
                num_val_vids = int(len(vids)*val_split)
                val_inds = np.random.permutation(range(len(vids)))[:num_val_vids]

            # split train-val
            if self.train:
                vids = [vids[i] for i in range(len(vids)) if i not in val_inds]
                labels = [labels[i] for i in range(len(labels)) if i not in val_inds]
                cal_mats = [cal_mats[i] for i in range(len(cal_mats)) if i not in val_inds]
                gape = [gape[i] for i in range(len(gape)) if i not in val_inds]
                plate = [plate[i] for i in range(len(plate)) if i not in val_inds]
            else:
                vids = [vids[i] for i in range(len(vids)) if i in val_inds]
                labels = [labels[i] for i in range(len(labels)) if i in val_inds]
                cal_mats = [cal_mats[i] for i in range(len(cal_mats)) if i in val_inds]
                gape = [gape[i] for i in range(len(gape)) if i in val_inds]
                plate = [plate[i] for i in range(len(plate)) if i in val_inds]

        # map index to vid/cam_images index
        self.index_to_img = [(i, j) for i, v in enumerate(vids) for j in range(len(v[0]))]
        self.vids = vids
        self.labels = labels
        self.cal_mats = cal_mats
        self.gape = gape
        self.plate = plate
        self.num_images = len(self.index_to_img)

        # calc fixed crop indices per video
        self.crop_inds_dict, self.crop = {}, None
        if fixed_crop_per_vid:
            self.crop = np_preprocess.CropLandmarks(ratio=fixed_crop_ratio, return_crop_point=True)
            for _vids, _labels in zip(self.vids, self.labels):
                for _vid, _label in zip(_vids, _labels):
                    _label = np.concatenate(_label, 0)
                    nonan_label = _label[~np.isnan(_label[:, 0]), :]
                    min_x, max_x = nonan_label[:, 0].min(), nonan_label[:, 0].max()
                    min_y, max_y = nonan_label[:, 1].min(), nonan_label[:, 1].max()
                    max_inds = [min_x, min_y, max_x, max_y]
                    vid_name = os.path.sep.join(_vid[0].split(os.path.sep)[-3:-1])
                    self.crop_inds_dict[vid_name] = max_inds

    def generate_target(self, landmarks, image_size, heatmap_size):
        """
        :param landmarks:  [num_joints, 2]. nan means no landmarks
        :param image_size: [3, H, W]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        image_size = np.array(image_size)
        heatmap_size = np.array(heatmap_size)

        num_landmarks = landmarks.shape[0]
        target_weight = np.ones((num_landmarks, 1), dtype=np.float32)
        target_weight[:, 0] = ~np.isnan(landmarks[:, 0])

        target = np.zeros((num_landmarks,
                           heatmap_size[1],
                           heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(num_landmarks):
            if target_weight[joint_id]:
                feat_stride = image_size / heatmap_size
                mu_x = int(landmarks[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(landmarks[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image
        imgs, labels, data = [], [], []
        vid_index, img_index = self.index_to_img[idx]
        for cam_idx in range(len(self.vids[vid_index])):
            img_name = self.vids[vid_index][cam_idx][img_index]
            if self.num_input_images > 1:
                n = int((self.num_input_images-1)/2)
                n_imgs = len(self.vids[vid_index][cam_idx])
                np_image = np.stack([np.array(Image.open(self.vids[vid_index][cam_idx][min(n_imgs-1, max(0, _idx))]).convert('L'))
                                     for _idx in range(img_index-n, img_index+n+1)], axis=2).astype(np.float32)
            else:
                pil_image = Image.open(img_name).convert('L')
                np_image = np.array(pil_image)
                np_image = np_image[..., np.newaxis].astype(np.float32)

            # get landmarks
            landmarks = self.labels[vid_index][cam_idx][img_index]
            if self.use_prev_hm_input:
                if img_index == 0:
                    hint_landmarks = np.nan * np.zeros(landmarks.shape, dtype=landmarks.dtype)
                else:
                    hint_landmarks = self.labels[vid_index][cam_idx][max(0, img_index-1)]
                landmarks = np.concatenate([landmarks, hint_landmarks], 0)

            crop_x, crop_y, scale = [None], [None], 0
            original_image_shape = np_image.shape[:2]

            current_vid_name = os.path.sep.join(img_name.split(os.path.sep)[-3:-1])
            crop_inds = self.crop_inds_dict.get(current_vid_name, False)
            if crop_inds:
                np_image, landmarks, (crop_x, crop_y) = self.crop(np_image, landmarks, inds=crop_inds)

            if self.np_transforms:
                for np_transform in self.np_transforms:
                    if isinstance(np_transform, np_preprocess.CropLandmarks):
                        np_image, landmarks, (crop_x, crop_y) = np_transform(np_image, landmarks)
                    elif isinstance(np_transform, np_preprocess.Resize):
                        np_image, landmarks, scale = np_transform(np_image, landmarks)
                    else:
                        np_image, landmarks = np_transform(np_image, landmarks)

                # remove landmarks out of frame
                invalid_inds = np.any(np.concatenate([landmarks < 0,
                                                      landmarks[:, 0:1] >= np_image.shape[1],
                                                      landmarks[:, 1:2] >= np_image.shape[0]], 1), axis=1)
                landmarks[invalid_inds, :] = np.nan

            if self.reutrn_target_as_heatmaps:
                target, target_weight = self.generate_target(landmarks, np_image.shape[:2], self.heatmap_size)
                target = torch.from_numpy(target)
                target_weight = torch.from_numpy(target_weight)
            else:
                # raise NotImplemented
                target, target_weight = torch.Tensor(landmarks), torch.Tensor(~np.isnan(landmarks[:, 0]))

            if self.transform:
                np_image = self.transform(np_image)

            imgs.append(np_image)
            labels.append(target)
            data.append({'img_name': img_name,
                         'scale': scale,
                         'crop_x': crop_x[0],
                         'crop_y': crop_y[0],
                         'img_size_x': original_image_shape[1],
                         'img_size_y': original_image_shape[0],
                         'is_gape': self.gape[vid_index][cam_idx][img_index],
                         'plate': self.plate[vid_index][cam_idx][img_index],
                         })

        cal_mats = torch.Tensor(self.cal_mats[vid_index])
        return imgs, labels, target_weight, cal_mats, data


if __name__ == '__main__':
    ds = Dataset2x2D(sigma=5, train=False, crop_landmarks=False, reutrn_target_as_heatmaps=False,
                     transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(ds, batch_size=4, shuffle=False)
    for i, (images, targets, target_weight, pp) in enumerate(train_loader):
        img = show_image_with_keypoints(Image.fromarray(np.uint8(images[0].cpu().detach().numpy()*255)[0,0,...]),
                                        targets[0].cpu().detach().numpy()[0, ...])
        img.show()
        img = show_image_with_keypoints(Image.fromarray(np.uint8(images[1].cpu().detach().numpy()*255)[0,0,...]),
                                        targets[1].cpu().detach().numpy()[0, ...])
        img.show()
