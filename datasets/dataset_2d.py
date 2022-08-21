
import os
from glob import glob
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2 as cv

from scripts.gen_2d_dataset import show_image_with_keypoints
import datasets.transforms as np_preprocess


class Dataset2D(Dataset):
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
                 imgs_folder="/data/projects/swat/users/eladc/project/dataset_2d_new",
                 label_file=None,
                 transform=None, np_transforms=None, reutrn_target_as_heatmaps=True, heatmap_size=(96, 96), sigma=1,
                 image_size=384, train=True, val_split=0.2, joints_weight=None, num_input_images=1,
                 use_prev_hm_input=False, fixed_crop_per_vid=False, fixed_crop_ratio=2.0):

        self.imgs_folder = imgs_folder
        imgs = glob(os.path.join(imgs_folder, "**", "*.jpg"))
        self.label_file = label_file if label_file else os.path.join(imgs_folder, "labels.pickle")
        with open(self.label_file, 'rb') as f:
            labels = pickle.load(f)
        self.transform = transform
        self.np_transforms = np_transforms
        self.reutrn_target_as_heatmaps = reutrn_target_as_heatmaps
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.image_size = image_size
        self.train = train
        self.use_different_joints_weight = True
        if joints_weight is None:
            joints_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.joints_weight = np.array(joints_weight*num_input_images).reshape((-1, 1))
        assert (num_input_images-1)/2 == int((num_input_images-1)/2)
        self.num_input_images = num_input_images
        assert not (use_prev_hm_input and num_input_images == 1)
        self.use_prev_hm_input = use_prev_hm_input

        if len(labels) != len(imgs):
            raise Exception('Mismatch between images and labels')

        self.imgs, self.labels = [], []
        for _label, _img in zip(labels, imgs):
            if not np.all(np.isnan(_label)):
                self.imgs.append(_img)
                self.labels.append(_label)

        # calc fixed crop indices per video
        self.crop_inds_dict, self.crop = {}, None
        if fixed_crop_per_vid:
            self.crop = np_preprocess.CropLandmarks(ratio=fixed_crop_ratio)
            last_vid_name = None
            # _inds = [0, 0, 0, 0]
            for img_name, _label in zip(self.imgs, self.labels):
                nonan_label = _label[~np.isnan(_label[:, 0]), :]
                min_x, max_x = nonan_label[:, 0].min(), nonan_label[:, 0].max()
                min_y, max_y = nonan_label[:, 1].min(), nonan_label[:, 1].max()
                max_inds = [min_x, min_y, max_x, max_y]
                current_vid_name = img_name.split(os.path.sep)[-2]
                if last_vid_name == current_vid_name:
                    self.crop_inds_dict[current_vid_name][:2] = list(np.minimum(self.crop_inds_dict[current_vid_name][:2], max_inds[:2]))
                    self.crop_inds_dict[current_vid_name][2:] = list(np.maximum(self.crop_inds_dict[current_vid_name][2:], max_inds[2:]))
                else:
                    self.crop_inds_dict[current_vid_name] = max_inds
                last_vid_name = current_vid_name

        if val_split is not None:
            if isinstance(val_split, (tuple, list)):
                vid_names = [f'vid_{i:03d}_' for i in val_split]
                if self.train:
                    inds = [i for i, img in enumerate(self.imgs)
                            if img.split(os.path.sep)[-2][:-4] not in vid_names]
                else:
                    inds = [i for i, img in enumerate(self.imgs)
                            if img.split(os.path.sep)[-2][:-4] in vid_names]
                self.imgs = [self.imgs[i] for i in inds]
                self.labels = [self.labels[i] for i in inds]
            else:
                # split train-val
                num_val_imgs = int(len(self.imgs)*val_split)
                if self.train:
                    self.imgs = self.imgs[num_val_imgs:]
                    self.labels = self.labels[num_val_imgs:]
                else:
                    self.imgs = self.imgs[:num_val_imgs]
                    self.labels = self.labels[:num_val_imgs]

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
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image
        img_name = self.imgs[idx]
        if self.num_input_images > 1:
            n = int((self.num_input_images-1)/2)
            vid_name = img_name.split(os.sep)[-2]
            img_idxs = [idx]
            for _idx in range(idx-1, idx-n-1, -1):
                _idx = max(0, _idx)
                if self.imgs[_idx].split(os.sep)[-2] == vid_name:
                    img_idxs.insert(0, _idx)
                else:
                    img_idxs.insert(0, img_idxs[0])
            for _idx in range(idx+1, idx+n+1):
                _idx = min(len(self.imgs)-1, _idx)
                if self.imgs[_idx].split(os.sep)[-2] == vid_name:
                    img_idxs.append(_idx)
                else:
                    img_idxs.append(img_idxs[-1])
            np_image = np.stack([np.array(Image.open(self.imgs[_idx]).convert('L'))
                                 for _idx in img_idxs], axis=2).astype(np.float32)
            # get landmarks
            landmarks = np.concatenate([self.labels[_idx] for _idx in img_idxs], axis=0).astype(np.float32)

        else:
            img_idxs = None
            pil_image = Image.open(img_name).convert('L')

            np_image = np.array(pil_image)
            np_image = np_image[..., np.newaxis].astype(np.float32)
            # get landmarks
            landmarks = self.labels[idx]

        if self.use_prev_hm_input:
            assert isinstance(img_idxs, list) and self.num_input_images == 3
            if img_idxs[0] == img_idxs[1]:
                hint_landmarks = np.nan * np.zeros(landmarks.shape, dtype=landmarks.dtype)
            else:
                hint_landmarks = self.labels[img_idxs[0]]
            landmarks = np.concatenate([landmarks, hint_landmarks], 0)

        current_vid_name = img_name.split(os.path.sep)[-2]
        crop_inds = self.crop_inds_dict.get(current_vid_name, False)
        if crop_inds:
            np_image, landmarks = self.crop(np_image, landmarks, inds=crop_inds)

        if self.np_transforms:
            for np_transform in self.np_transforms:
                np_image, landmarks = np_transform(np_image, landmarks)
                # print(np_transform, np_image.dtype)

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

        return np_image, target, target_weight, [img_name]


if __name__ == '__main__':
    ds = Dataset2D(imgs_folder=r'C:\eladc\project\dataset_2d_new',
                   sigma=2, transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(127.5, 127.5)]),
                   np_transforms=[np_preprocess.FlipLeftRight(),
                                  np_preprocess.CropLandmarks(ratio=3.0),
                                  np_preprocess.RandomRotateScale(max_angle=40, scale_limits=[0.7, 1.5]),
                                  np_preprocess.RandomSquash(100),
                                  np_preprocess.CropLandmarks(ratio=1.3, random_crop=True),
                                  np_preprocess.Resize(image_size=(384, 384)),
                                  np_preprocess.CutOut(cutout_prob=0.9,
                                                       min_cutout=20, max_cutout=100)],
                   reutrn_target_as_heatmaps=False)
    train_loader = DataLoader(ds, batch_size=16, shuffle=True)
    for i, (image, target, target_weight, pp) in enumerate(train_loader):
        print(f'Batch {i+1} of {len(train_loader)}')
        for img_index in range(image.size(0)):
            img = show_image_with_keypoints(np.uint8(image.detach().cpu().numpy()[img_index, 0, :, :]*127.5+127.5),
                                            target.detach().cpu().numpy()[img_index, ...])
            img.save(f'/data/projects/sw_results/eladc/project/fish_train_images/{i}_{img_index}.jpg')
        # img.show()
        a=1
