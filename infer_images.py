
import time
from datetime import datetime

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2

from datasets.dataset_2d import Dataset2D

from test import parse_args
from lib.config import cfg
from lib.config import update_config
import lib.models as models
from lib.core.function import AverageMeter, accuracy
from datasets.dataset_2d import show_image_with_keypoints


def build_big_img(img_size, img_with_landmarks, heatmaps, out):
    # build large image
    big_img = np.zeros((img_size[0]*3+2, img_size[1]*4+3, 3))
    # borders
    big_img[img_size[0], :, 1] = 255
    big_img[2*img_size[0]+1, :, 1] = 255
    big_img[:, img_size[1], 1] = 255
    big_img[:, 2*img_size[1]+1, 1] = 255
    big_img[:, 3*img_size[1]+2, 1] = 255
    # image with landmarks
    big_img[:img_size[0], :img_size[1], :] = np.array(img_with_landmarks)
    # heatmaps
    big_img[img_size[0]+1:2*img_size[0]+1, :img_size[1], 2] = np.array(heatmaps[0])[..., 0]
    big_img[2*img_size[0]+2:, :img_size[1], 2] = np.array(heatmaps[1])[..., 0]

    big_img[:img_size[0], img_size[1]+1:2*img_size[1]+1, 2] = np.array(heatmaps[2])[..., 0]
    big_img[img_size[0]+1:2*img_size[0]+1, img_size[1]+1:2*img_size[1]+1, 2] = np.array(heatmaps[3])[..., 0]
    big_img[2*img_size[0]+2:, img_size[1]+1:2*img_size[1]+1, 2] = np.array(heatmaps[4])[..., 0]

    big_img[:img_size[0], 2*img_size[1]+2:3*img_size[1]+2, 2] = np.array(heatmaps[5])[..., 0]
    big_img[img_size[0]+1:2*img_size[0]+1, 2*img_size[1]+2:3*img_size[1]+2, 2] = np.array(heatmaps[6])[..., 0]
    big_img[2*img_size[0]+2:, 2*img_size[1]+2:3*img_size[1]+2, 2] = np.array(heatmaps[7])[..., 0]

    big_img[:img_size[0], 3*img_size[1]+3:, 2] = np.array(heatmaps[8])[..., 0]
    big_img[img_size[0]+1:2*img_size[0]+1, 3*img_size[1]+3:, 2] = np.array(heatmaps[9])[..., 0]
    # big_img[2*img_size[0]+2:, 3*img_size[1]+3:, 2] =
    # outputs
    big_img[img_size[0]+1:2*img_size[0]+1, :img_size[1], 0] = 255*cv2.resize(out[0, ...], img_size)
    big_img[2*img_size[0]+2:, :img_size[1], 0] = 255*cv2.resize(out[1, ...], img_size)

    big_img[:img_size[0], img_size[1]+1:2*img_size[1]+1, 0] = 255*cv2.resize(out[2, ...], img_size)
    big_img[img_size[0]+1:2*img_size[0]+1, img_size[1]+1:2*img_size[1]+1, 0] = 255*cv2.resize(out[3, ...], img_size)
    big_img[2*img_size[0]+2:, img_size[1]+1:2*img_size[1]+1, 0] = 255*cv2.resize(out[4, ...], img_size)

    big_img[:img_size[0], 2*img_size[1]+2:3*img_size[1]+2, 0] = 255*cv2.resize(out[5, ...], img_size)
    big_img[img_size[0]+1:2*img_size[0]+1, 2*img_size[1]+2:3*img_size[1]+2, 0] = 255*cv2.resize(out[6, ...], img_size)
    big_img[2*img_size[0]+2:, 2*img_size[1]+2:3*img_size[1]+2, 0] = 255*cv2.resize(out[7, ...], img_size)

    big_img[:img_size[0], 3*img_size[1]+3:, 0] = 255*cv2.resize(out[8, ...], img_size)
    big_img[img_size[0]+1:2*img_size[0]+1, 3*img_size[1]+3:, 0] = 255*cv2.resize(out[9, ...], img_size)
    # big_img[2*img_size[0]+2:, 3*img_size[1]+3:, 2] =

    return Image.fromarray(np.uint8(big_img))


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    output_dir = "/data/path/results"
    if cfg.MODEL.NAME.lower() == 'lpn':
        get_model = models.lpn.get_pose_net
    else:
        raise NotImplemented
    model = get_model(cfg, is_train=False).cuda()
    model.load_state_dict(torch.load('/data/path/state_dict.pth'))

    VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]
    if False:  # train
        is_train = True
        crop_ratio = 3.0
    else:
        is_train = False
        crop_ratio = 2.4
    reutrn_target_as_heatmaps = False
    dataset = Dataset2D(reutrn_target_as_heatmaps=reutrn_target_as_heatmaps,
                        transform=transforms.Compose([transforms.ToTensor()]),
                        crop_landmarks=True, train=is_train, val_split=VAL_SPLIT,
                        crop_ratio=crop_ratio)

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    comment = '' if args.comment is None else ' '.join(args.comment)
    time_stamp = f'output_imgs{os.path.sep}{datetime.now().strftime("%Y%m%d_%H%M%S")}__{comment.replace(" ", "_")}'
    results_folder = os.path.join(os.environ.get('SWAT'), 'results', os.environ.get('USER'),
                                  'fish_landmarks', f'{"train_" if is_train else "eval_"}' + time_stamp)
    print('results folder:', results_folder)
    val_acc = AverageMeter()

    os.makedirs(results_folder)
    model.eval()
    with torch.no_grad():
        for i, (data, target, target_weight, pp) in enumerate(data_loader):
            print(f'Inferring batch #{i} of {len(data_loader)}')
            # compute output
            output = model(data.cuda())

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if False:  # reutrn_target_as_heatmaps:
                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
                val_acc.update(avg_acc, cnt)
            else:
                for j in range(data.size(0)):
                    _output = np.zeros((10, 2))
                    out = output.cpu().detach().numpy()[j, ...]
                    for l in range(out.shape[0]):
                        if target_weight.cpu().detach().numpy()[j, l] == 1:
                            _output[l, 0] = 4*np.argmax(np.amax(out[l, ...], axis=0))
                            _output[l, 1] = 4*np.argmax(np.amax(out[l, ...], axis=1))
                        else:
                            _output[l, 0] = np.nan
                            _output[l, 1] = np.nan
                    if reutrn_target_as_heatmaps:
                        _label = np.zeros((10, 2))
                        np_target = target.cpu().detach().numpy()[j, ...]
                        for l in range(np_target.shape[0]):
                            if target_weight.cpu().detach().numpy()[j, l] == 1:
                                _label[l, 0] = 4 * np.argmax(np.amax(np_target[l, ...], axis=0))
                                _label[l, 1] = 4 * np.argmax(np.amax(np_target[l, ...], axis=1))
                            else:
                                _label[l, 0] = np.nan
                                _label[l, 1] = np.nan
                        img_with_landmarks = show_image_with_keypoints(np.uint8(data.cpu().detach().numpy()[j, 0, ...] * 255),
                                                                       np.concatenate([_output, _label], 0))
                        img_size = img_with_landmarks.size
                        heatmaps = [Image.fromarray(np.int8(255*hm)).convert('RGB').resize(img_size)
                                    for hm in target.cpu().detach().numpy()[j, ...]]

                        # build large image
                        img = build_big_img(img_size, img_with_landmarks, heatmaps, out)
                        img.save(os.path.join(results_folder, f'img_{i*data.size(0)+j:05d}.jpg'))
                    else:
                        _label = target.cpu().detach().numpy()[j, ...]
                        img = show_image_with_keypoints(np.uint8(data.cpu().detach().numpy()[j, 0, ...] * 255),
                                                        np.concatenate([_output, _label], 0))
                        img.save(os.path.join(results_folder, f'img_{i*data.size(0)+j:05d}.jpg'))

    if reutrn_target_as_heatmaps:
        print(f'Validation accuracy: {val_acc.avg:.5f}')
    else:
        print('images saved at', results_folder)
