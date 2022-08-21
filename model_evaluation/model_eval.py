import pickle
from datetime import datetime

import time
import os
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import conv2d
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from model_evaluation.utils import calc_ls, calc_ransac
from datasets.dataset_2x2d import Dataset2x2D
import datasets.transforms as np_transforms

from test import parse_args
from lib.config import cfg
from lib.config import update_config
import lib.models as models
from datasets.dataset_2d import show_image_with_keypoints
import model_evaluation.epipolar_geometry_utils as egu
from model_evaluation.utils import calc_characteristics, calc_max_inds, calc_local_mean_no_outliers, \
    load_landmark_diff_stats, FinalParams


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

# base vector: 8/9 -> 4
# head angle: 0 -> 2
# pectoral angle: 8/9 -> 6/7
# tail angle: 4 -> 5


class HeatmapFilter(torch.nn.Module):
    def __init__(self, stat_mat, heatmap_scale, g_width=6, mode='gaussian'):
        if mode not in ['gaussian', 'uniform']:
            raise Exception(f'Unknown HeatmapFilter mode: {mode}')
        super(HeatmapFilter, self).__init__()
        self.g_width = g_width
        self.mode = mode
        self.heatmap_scale = heatmap_scale
        x = np.arange(2 * g_width + 1) - g_width
        y = x[:, np.newaxis]
        # self.heatmap_filter = (x ** 2 + y ** 2).reshape((2 * g_width + 1, 2 * g_width + 1, 1)) /
        #                              (2 * (stat_mat[:, 0, 0]*2).reshape((1, 1, -1)) ** 2))
        if self.mode == 'gaussian':
            self.heatmap_filter = np.exp(- (x ** 2 + y ** 2).reshape((2 * g_width + 1, 2 * g_width + 1, 1)) /
                                         (2 * (stat_mat[:, 0, 0]*2).reshape((1, 1, -1)) ** 2))
            self.heatmap_filter = self.heatmap_filter[np.newaxis].transpose((3, 0, 1, 2))
            num_landmarks = self.heatmap_filter.shape[0]
            self.conv2d = torch.nn.Conv2d(num_landmarks, num_landmarks, 2 * g_width + 1,
                                          padding=g_width, groups=num_landmarks, bias=False)
            self.conv2d.weight.data = torch.Tensor(self.heatmap_filter.astype(np.float32))
        else:
            self.heatmap_filter = (x ** 2 + y ** 2).reshape((2 * g_width + 1, 2 * g_width + 1, 1)) < (stat_mat[:, 0, 0]*10).reshape((1, 1, -1)) ** 2

    def forward(self, hm):
        if self.mode == 'gaussian':
            t_hm = torch.Tensor(hm[np.newaxis, ...])
            expected_hm = self.conv2d(t_hm).cpu().detach().numpy()[0, ...]
        else:
            max_inds = calc_max_inds(hm, np.ones(10), None)
            expected_hm = np.zeros(hm.shape)
            for l in range(hm.shape[0]):
                if np.all(max_inds[l, ...] != 0):
                    sx = slice(int(max_inds[l, 0])-self.g_width, int(max_inds[l, 0])+self.g_width+1)
                    sy = slice(int(max_inds[l, 1])-self.g_width, int(max_inds[l, 1])+self.g_width+1)
                    expected_hm[l, sy, sx] = self.heatmap_filter[:, :, l]
        return expected_hm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), \
           sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    output_dir = "/data/projects/swat/users/eladc/project/results"
    if cfg.MODEL.NAME.lower() == 'lpn':
        get_model = models.lpn.get_pose_net
    elif cfg.MODEL.NAME.lower() == 'pose_hrnet':
        get_model = models.pose_hrnet.get_pose_net
    else:
        raise NotImplemented
    model = get_model(cfg, is_train=False).cuda()

    run_name = ''
    if args.comment:
        model_file = args.comment[0]
        if len(args.comment) == 2:
            run_name = args.comment[1]
    else:
        raise Exception('Missing model file')
    model.load_state_dict(torch.load(model_file))   # model.init_weights(model_file)
    output_dir = os.path.join(output_dir, model_file.split(os.sep)[-2] + run_name)
    print(f'Saving model evaluation files to {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    stat_mat = load_landmark_diff_stats()

    inds_dict = {'vid_000': [13, 25, 37],
                 'vid_004': [4, 20, 36],
                 'vid_012': [85, 103, 124],
                 'vid_016': [26, 38, 62],
                 'vid_021': [13, 51, 95],
                 'vid_026': [47, 72, 96],
                 'vid_033': [7, 49, 71],
                 'vid_039': [32, 62, 94],
                 }

    # VAL_SPLIT = list(range(41))
    VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]
    # VAL_SPLIT = [4, 12, 16, 21, 26, 33, 39]
    # VAL_SPLIT = [16]
    crop_ratio = cfg.DATASET.VAL_CROP_RATIO #2.4
    dataset = Dataset2x2D(train=False, val_split=VAL_SPLIT, reutrn_target_as_heatmaps=False,
                          num_input_images = cfg.MODEL.NUM_INPUT_IMAGES,
                          image_size=cfg.MODEL.IMAGE_SIZE[0], heatmap_size=cfg.MODEL.HEATMAP_SIZE,
                          use_prev_hm_input=cfg.MODEL.USE_PREV_HM_INPUT,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(127.5, 127.5)]),
                          np_transforms=[#np_transforms.CropLandmarks(ratio=crop_ratio, return_crop_point=True),
                                         np_transforms.Resize(image_size=cfg.MODEL.IMAGE_SIZE, return_scale=True),
                                         ],
                          fixed_crop_per_vid=True, fixed_crop_ratio=crop_ratio)

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    p = 400  # pad resized heatmaps

    save_images = True
    refine = True
    filter_outputs = False
    if filter_outputs:
        hmf = HeatmapFilter(stat_mat, dataset.image_size / dataset.heatmap_size[0], g_width=30, mode='uniform')
    vid_name, image_index = 'None', -1
    test_flip_lr = False
    res = []
    model_output_2d, labels_2d = {}, {}
    model_output_3d, model_output_3d_dlt, labels_3d, is_gape, plate = {}, {}, {}, {}, {}
    heatmaps_list1, heatmaps_list2 = [], []
    heatmaps_depth = 5
    landmarks_dists_mat = []
    run_geometric = False
    dlt_geometric_diff = []

    use_real_hm = False
    prev_hm1, prev_hm2 = None, None
    per_image = False

    model.eval()
    for i, (data, target, target_weight, cal_mats, pp) in enumerate(data_loader):
        print(f'Inferring batch #{i+1} of {len(data_loader)}')
        if not per_image:
            # compute output
            with torch.no_grad():
                if cfg.MODEL.USE_PREV_HM_INPUT:
                    input_target1 = target[0].cuda()[:, 10:, ...]
                    input_target2 = target[1].cuda()[:, 10:, ...]
                    output1 = model(data[0].cuda(), input_target1).cpu().detach().numpy()
                    output2 = model(data[1].cuda(), input_target2).cpu().detach().numpy()
                else:
                    output1 = model(data[0].cuda()).cpu().detach().numpy()
                    output2 = model(data[1].cuda()).cpu().detach().numpy()
            if test_flip_lr:
                if cfg.MODEL.USE_PREV_HM_INPUT:
                    raise NotImplemented
                with torch.no_grad():
                    output1_flipped = model(torch.flip(data[0].cuda(), [3])).cpu().detach().numpy()
                    output2_flipped = model(torch.flip(data[1].cuda(), [3])).cpu().detach().numpy()
                output1_flipped_back = output1_flipped[:, dataset._LANDMARK_FLIP, :, ::-1]
                output2_flipped_back = output2_flipped[:, dataset._LANDMARK_FLIP, :, ::-1]
                output1 = (output1 + output1_flipped_back) / 2
                output2 = (output2 + output2_flipped_back) / 2

        if cfg.MODEL.USE_PREV_HM_INPUT:
            target1 = target[0].cpu().detach().numpy()[:, :10, ...]
            target2 = target[1].cpu().detach().numpy()[:, :10, ...]
            target_weight = target_weight.cpu().detach().numpy()[:, :10]
        else:
            target1 = target[0].cpu().detach().numpy()
            target2 = target[1].cpu().detach().numpy()
            target_weight = target_weight.cpu().detach().numpy()
        cal_mats = cal_mats.cpu().detach().numpy()

        for img_idx in range(target1.shape[0]):
            _vid_name = pp[0]['img_name'][img_idx].split(os.sep)[-3]
            if vid_name == _vid_name:
                image_index = image_index + 1
            else:
                prev_hm1, prev_hm2 = None, None
                image_index = 0
            vid_name = _vid_name

            if per_image:
                # compute output
                with torch.no_grad():
                    if cfg.MODEL.USE_PREV_HM_INPUT:
                        if image_index == 0:
                            # at first image of video use current image labels
                            input_target1 = target[0][img_idx:img_idx + 1, :10, ...]
                            input_target2 = target[1][img_idx:img_idx + 1, :10, ...]
                        else:
                            # use previous image labels\outputs
                            if use_real_hm:
                                input_target1, input_target2 = torch.Tensor(prev_hm1), torch.Tensor(prev_hm2)
                            else:
                                input_target1 = target[0][img_idx:img_idx + 1, 10:, ...]
                                input_target2 = target[1][img_idx:img_idx + 1, 10:, ...]
                        output1 = model(data[0][img_idx:img_idx+1, ...].cuda(), input_target1.cuda()).cpu().detach().numpy()
                        output2 = model(data[1][img_idx:img_idx+1, ...].cuda(), input_target2.cuda()).cpu().detach().numpy()

                        if per_image:
                            output_inds1 = calc_max_inds(output1[0, ...], target_weight[img_idx, :], list(data[0].shape)[2:])
                            prev_hm1 = dataset.generate_target(output_inds1, dataset.image_size, dataset.heatmap_size)[0][np.newaxis, ...]

                            for _l in range(10):
                                ii = Image.fromarray(np.uint8(np.stack([cv2.resize(output1[0, _l, ...], (384, 384)),
                                                                        cv2.resize(prev_hm1[0, _l, ...], (384, 384)),
                                                                        data[0][img_idx, 1, ...] / 2 + 0.5], 2) * 255))
                                os.makedirs(os.path.join(output_dir, vid_name + '_debug', f'cam0_landmark{_l}'), exist_ok=True)
                                ii.save(os.path.join(output_dir, vid_name + '_debug', f'cam0_landmark{_l}', f'{image_index}.jpg'))

                            output_inds2 = calc_max_inds(output2[0, ...], target_weight[img_idx, :], list(data[0].shape)[2:])
                            prev_hm2 = dataset.generate_target(output_inds2, dataset.image_size, dataset.heatmap_size)[0][np.newaxis, ...]
                    else:
                        output1 = model(data[0][img_idx:img_idx+1, ...].cuda()).cpu().detach().numpy()
                        output2 = model(data[1][img_idx:img_idx+1, ...].cuda()).cpu().detach().numpy()

            if vid_name not in model_output_3d:
                heatmaps_list1, heatmaps_list2 = [], []

            ii = 0 if per_image else img_idx
            output_inds1 = calc_max_inds(output1[ii, ...], target_weight[img_idx, :], list(data[0].shape)[2:])
            output_inds2 = calc_max_inds(output2[ii, ...], target_weight[img_idx, :], list(data[1].shape)[2:])

            if cfg.MODEL.USE_PREV_HM_INPUT:
                _target1 = calc_max_inds(target1[img_idx, ...], target_weight[img_idx, :], list(data[0].shape)[2:])
                _target2 = calc_max_inds(target2[img_idx, ...], target_weight[img_idx, :], list(data[1].shape)[2:])
            else:
                _target1 = target1[img_idx, ...]
                _target2 = target2[img_idx, ...]

            if save_images:
                os.makedirs(os.path.join(output_dir, vid_name), exist_ok=True)

                np_image = data[0][img_idx, 0, ...].detach().cpu().numpy()
                i1 = show_image_with_keypoints(Image.fromarray((np_image-np_image.min())/(np_image.max()-np_image.min())*255),
                                               np.concatenate([output_inds1, _target1], 0))
                np_image = data[1][img_idx, 0, ...].detach().cpu().numpy()
                i2 = show_image_with_keypoints(Image.fromarray((np_image-np_image.min())/(np_image.max()-np_image.min())*255),
                                               np.concatenate([output_inds2, _target2], 0))
                dst = Image.new('RGB', (i1.width + i2.width, i1.height), (0, 0, 0))
                dst.paste(i1, (0, 0))
                dst.paste(i2, (i1.width, 0))
                dst.save(os.path.join(output_dir, vid_name, f'{image_index}.jpg'))

            # switch to original coordinates
            crop_x1 = pp[0]['crop_x'][img_idx].numpy()
            crop_y1 = pp[0]['crop_y'][img_idx].numpy()
            crop_x2 = pp[1]['crop_x'][img_idx].numpy()
            crop_y2 = pp[1]['crop_y'][img_idx].numpy()
            scale1 = pp[0]['scale'][img_idx].numpy()
            scale2 = pp[1]['scale'][img_idx].numpy()
            img1_size_y = pp[0]['img_size_y'][img_idx].numpy()
            img1_size_x = pp[0]['img_size_x'][img_idx].numpy()
            img2_size_y = pp[1]['img_size_y'][img_idx].numpy()
            img2_size_x = pp[1]['img_size_x'][img_idx].numpy()

            _plate1 = pp[0]['plate'][img_idx].numpy()
            _plate2 = pp[1]['plate'][img_idx].numpy()

            output_inds1 = output_inds1 / scale1 + np.array([[crop_x1, crop_y1]])
            output_inds2 = output_inds2 / scale2 + np.array([[crop_x2, crop_y2]])
            landmarks1 = _target1 / scale1 + np.array([[crop_x1, crop_y1]])
            landmarks2 = _target2 / scale2 + np.array([[crop_x2, crop_y2]])

            # pil_img1 = Image.open(pp[0]['img_name'][img_idx])
            # show_image_with_keypoints(pil_img1, np.concatenate([output_inds1, landmarks1, _plate1.reshape((1,2))], 0)).show()
            # pil_img2 = Image.open(pp[1]['img_name'][img_idx])
            # show_image_with_keypoints(pil_img2, np.concatenate([output_inds2, landmarks2, _plate2.reshape((1,2))], 0)).show()

            _landmarks1 = landmarks1.copy()
            _landmarks2 = landmarks2.copy()
            _landmarks1[:, 1] = img1_size_y - _landmarks1[:, 1]
            _landmarks2[:, 1] = img2_size_y - _landmarks2[:, 1]

            _plate1[1] = img1_size_y - _plate1[1]
            _plate2[1] = img2_size_y - _plate2[1]

            if filter_outputs: # project heatmap on original image shape before indices calculation
                fullsize = (int(img1_size_y + 2*p), int(img1_size_x + 2*p))
                heatmap_resize_shape = tuple(np.round(output1[img_idx, ...].shape[1:] / scale1 * hmf.heatmap_scale).astype(np.int32))
                fullsize_heatmap1 = np.zeros(output1[img_idx, ...].shape[:1] + fullsize, dtype=np.float32)
                sy = slice(p+int(crop_y1), p + int(crop_y1) + heatmap_resize_shape[0])
                sx = slice(p + int(crop_x1), p + int(crop_x1) + heatmap_resize_shape[1])
                fullsize_heatmap1[:, sy, sx] = cv2.resize(output1[img_idx, ...].transpose((1, 2, 0)),
                                                          heatmap_resize_shape,
                                                          interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                if len(heatmaps_list1) > 0:
                    expected_heatmap = hmf(heatmaps_list1[-1])

                    img = np.array(Image.open(pp[0]['img_name'][img_idx]))
                    for _l in range(10):
                        img[:, :, 1] = np.uint8(expected_heatmap[_l, p:-p, p:-p]*255)
                        img[:, :, 2] = np.uint8(fullsize_heatmap1[_l, p:-p, p:-p]/fullsize_heatmap1[0, ...].max()*255)
                        os.makedirs(os.path.join(output_dir, vid_name+'_debug', f'cam0_landmark{_l}'), exist_ok=True)
                        Image.fromarray(img).save(os.path.join(output_dir, vid_name+'_debug', f'cam0_landmark{_l}', f'{image_index}.jpg'))

                    fullsize_heatmap1 = expected_heatmap * fullsize_heatmap1
                _output_inds1 = calc_max_inds(fullsize_heatmap1, target_weight[img_idx, :], None) - p
                # show_image_with_keypoints(Image.open(pp[0]['img_name'][img_idx]), output_inds1).show()

                heatmap_resize_shape = tuple(np.round(output2[img_idx, ...].shape[1:] / scale2 * hmf.heatmap_scale).astype(np.int32))
                fullsize_heatmap2 = np.zeros(output2[img_idx, ...].shape[:1] + fullsize, dtype=np.float32)
                sy = slice(p + int(crop_y2), p + int(crop_y2) + heatmap_resize_shape[0])
                sx = slice(p + int(crop_x2), p + int(crop_x2) + heatmap_resize_shape[1])
                fullsize_heatmap2[:, sy, sx] = cv2.resize(output2[img_idx, ...].transpose((1, 2, 0)),
                                                          heatmap_resize_shape,
                                                          interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                if len(heatmaps_list2) > 0:
                    expected_heatmap = hmf(heatmaps_list2[-1])

                    img = np.array(Image.open(pp[1]['img_name'][img_idx]))
                    for _l in range(10):
                        img[:, :, 1] = np.uint8(expected_heatmap[_l, p:-p, p:-p]*255)
                        img[:, :, 2] = np.uint8(fullsize_heatmap2[_l, p:-p, p:-p]/fullsize_heatmap2[0, ...].max()*255)
                        os.makedirs(os.path.join(output_dir, vid_name+'_debug', f'cam1_landmark{_l}'), exist_ok=True)
                        Image.fromarray(img).save(os.path.join(output_dir, vid_name+'_debug', f'cam1_landmark{_l}', f'{image_index}.jpg'))

                    fullsize_heatmap2 = expected_heatmap * fullsize_heatmap2
                _output_inds2 = calc_max_inds(fullsize_heatmap2, target_weight[img_idx, :], None) - p
                # show_image_with_keypoints(Image.open(pp[1]['img_name'][img_idx]), output_inds2).show()

                if save_images:
                    os.makedirs(os.path.join(output_dir, vid_name+'_fullsize'), exist_ok=True)

                    pil_image = Image.open(pp[0]['img_name'][img_idx])
                    i1 = show_image_with_keypoints(pil_image,
                                                   np.concatenate([output_inds1, _output_inds1], 0))
                    pil_image = Image.open(pp[1]['img_name'][img_idx])
                    i2 = show_image_with_keypoints(pil_image,
                                                   np.concatenate([output_inds2, _output_inds2], 0))
                    dst = Image.new('RGB', (i1.width + i2.width, i1.height), (0, 0, 0))
                    dst.paste(i1, (0, 0))
                    dst.paste(i2, (i1.width, 0))
                    dst.save(os.path.join(output_dir, vid_name+'_fullsize', f'{image_index}.jpg'))

                output_inds1, output_inds2 = _output_inds1, _output_inds2

            label_3d, _ = egu.calc_3d_from_2x2d(_landmarks1, _landmarks2, cal_mats[img_idx, ...], geometric=False)
            plate_3d, _ = egu.calc_3d_from_2x2d(_plate1.reshape((1, 2)), _plate2.reshape((1, 2)), cal_mats[img_idx, ...], geometric=False)

            # update landmarks_dists_mat
            _dist_mat = np.zeros((label_3d.shape[0], label_3d.shape[0]))
            for j in range(label_3d.shape[0]):
                for k in range(label_3d.shape[0]):
                    _dist_mat[j, k] = np.linalg.norm(label_3d[j, ...] - label_3d[k, ...])
            landmarks_dists_mat.append(_dist_mat)

            _output_inds1 = output_inds1.copy()
            _output_inds2 = output_inds2.copy()
            _output_inds1[:, 1] = img1_size_y - _output_inds1[:, 1]
            _output_inds2[:, 1] = img2_size_y - _output_inds2[:, 1]
            out_3d, dlt_output = egu.calc_3d_from_2x2d(_output_inds1, _output_inds2, cal_mats[img_idx, ...], geometric=run_geometric)

            dlt_geometric_diff.append(np.linalg.norm(dlt_output - label_3d, axis=1) -
                                      np.linalg.norm(out_3d - label_3d, axis=1))
            res.append(np.linalg.norm(out_3d - label_3d, axis=1))

            # collect results
            if vid_name in model_output_3d:
                if filter_outputs:
                    heatmaps_list1.append(fullsize_heatmap1)
                    if len(heatmaps_list1) > heatmaps_depth:
                        heatmaps_list1.pop(0)
                    heatmaps_list2.append(fullsize_heatmap2)
                    if len(heatmaps_list2) > heatmaps_depth:
                        heatmaps_list2.pop(0)
                model_output_2d[vid_name].extend([_output_inds1, _output_inds2])
                labels_2d[vid_name].extend([_landmarks1, _landmarks2])
                model_output_3d[vid_name].append(out_3d)
                model_output_3d_dlt[vid_name].append(dlt_output)
                is_gape[vid_name].append(pp[0]['is_gape'][img_idx].numpy())
                plate[vid_name].append(plate_3d)
                labels_3d[vid_name].append(label_3d)
            else:
                if filter_outputs:
                    heatmaps_list1 = [fullsize_heatmap1]
                    heatmaps_list2 = [fullsize_heatmap2]
                model_output_2d[vid_name] = [_output_inds1, _output_inds2]
                labels_2d[vid_name] = [_landmarks1, _landmarks2]
                model_output_3d[vid_name] = [out_3d]
                model_output_3d_dlt[vid_name] = [dlt_output]
                is_gape[vid_name] = [pp[0]['is_gape'][img_idx].numpy()]
                plate[vid_name] = [plate_3d]
                labels_3d[vid_name] = [label_3d]

    with open(os.path.join(output_dir, 'data.pickle'), 'wb') as f:
        pickle.dump({'model_output_3d': model_output_3d,
                     'model_output_3d_dlt': model_output_3d_dlt,
                     'labels_3d': labels_3d,
                     'model_output_2d': model_output_2d,
                     'labels_2d': labels_2d,
                     'dlt_geometric_diff': dlt_geometric_diff,
                     'res': res,
                     }, f)

    if True:  # generate performance report
        print('Writing Report')
        _data = {}
        for vid_name in labels_2d:

            # 2D
            _label = np.stack(labels_2d[vid_name])
            _out = np.stack(model_output_2d[vid_name])
            _diff = np.linalg.norm(_label-_out, axis=2)

            fig, axs = plt.subplots(3, 4, figsize=(15, 10))
            fig.suptitle(f'{vid_name} 2d error histograms [pixel]')
            hist_range = (0, 50)

            values_2d = _diff.flatten()[~np.isnan(_diff.flatten())]
            mean_2d, std_2d = values_2d.mean(), values_2d.std()
            hist, bin_edges = np.histogram(values_2d, bins=100, range=hist_range)
            axs[2, 3].plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist)
            axs[2, 3].set_title('all landmarks')
            axs[2, 3].grid()

            stat_2d_per_landmark = -np.ones((_diff.shape[1], 2))
            for landmark_idx in range(_diff.shape[1]):
                _values = _diff[~np.isnan(_diff[:, landmark_idx]), landmark_idx]
                if len(_values):
                    stat_2d_per_landmark[landmark_idx, :] = np.array([_values.mean(), _values.std()])
                    _values[_values>hist_range[1]] = hist_range[1]
                    hist, bin_edges = np.histogram(_values, bins=100, range=hist_range)

                    i, j = int(landmark_idx/4), landmark_idx % 4
                    axs[i, j].plot((bin_edges[:-1] + bin_edges[1:])/ 2, hist)
                    axs[i, j].set_title(f'#{landmark_idx}: mean={stat_2d_per_landmark[landmark_idx, 0]: 2.2f},  std={stat_2d_per_landmark[landmark_idx, 1]: 2.2f}')
                    axs[i, j].grid()

            fig.savefig(os.path.join(output_dir, f'hist_diff2d_{vid_name}.png'))
            plt.close()

            # 3D
            _label = np.stack(labels_3d[vid_name])
            _out = np.stack(model_output_3d[vid_name])
            _diff = np.linalg.norm(_label-_out, axis=2)

            fig, axs = plt.subplots(3, 4, figsize=(15, 10))
            fig.suptitle(f'{vid_name} 3d error histograms')
            hist_range = (0, 5)

            values_3d = _diff.flatten()[~np.isnan(_diff.flatten())]
            mean_3d, std_3d = values_3d.mean(), values_3d.std()
            hist, bin_edges = np.histogram(values_3d, bins=100, range=hist_range)
            axs[2, 3].plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist)
            axs[2, 3].set_title('all landmarks')
            axs[2, 3].grid()

            stat_3d_per_landmark = -np.ones((_diff.shape[1], 2))
            for landmark_idx in range(_diff.shape[1]):
                _values = _diff[~np.isnan(_diff[:, landmark_idx]), landmark_idx]
                if len(_values):
                    stat_3d_per_landmark[landmark_idx, :] = np.array([_values.mean(), _values.std()])
                    _values[_values>hist_range[1]] = hist_range[1]
                    hist, bin_edges = np.histogram(_values, bins=100, range=hist_range)

                    i, j = int(landmark_idx/4), landmark_idx % 4
                    axs[i, j].plot((bin_edges[:-1] + bin_edges[1:])/ 2, hist)
                    axs[i, j].set_title(f'#{landmark_idx}: mean={stat_3d_per_landmark[landmark_idx, 0]: 2.2f},  std={stat_3d_per_landmark[landmark_idx, 1]: 2.2f}')
                    axs[i, j].grid()

            fig.savefig(os.path.join(output_dir, f'hist_diff3d_{vid_name}.png'))
            plt.close()

            _data[vid_name] = {'all_2d': [mean_2d, std_2d],
                               '2d': stat_2d_per_landmark,
                               'all_3d': [mean_3d, std_3d],
                               '3d': stat_3d_per_landmark,
                               }

        with open(os.path.join(output_dir, 'report.pickle'), 'wb') as f:
            pickle.dump(_data, f)

    if True and run_geometric:  # plot diff scatter-plots between geometric and DLT
        stacked_res = np.stack(res)
        stacked_diff = np.stack(dlt_geometric_diff)
        th = 0.2
        dists = []
        for landmark_idx in range(stacked_diff.shape[1]):
            values = stacked_diff[~np.isnan(stacked_diff[:, landmark_idx]), landmark_idx]
            res_values = stacked_res[~np.isnan(stacked_res[:, landmark_idx]), landmark_idx]

            fig, axs = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'Landmark {landmark_idx}: Estimation method error vs. estimation error')
            axs[0].scatter(res_values, values, marker='x')
            axs[0].set_xlabel('output error: dist(label, geometric) [cm]')
            axs[0].set_ylabel('est error: DLT error - geometric error [cm]')
            axs[0].grid()
            axs[0].set_title(f'Landmark {landmark_idx}: Estimation method error vs. estimation error')
            axs[1].scatter(res_values[np.abs(values)<th], values[np.abs(values)<th], marker='x')
            axs[1].set_xlabel('output error: dist(label, geometric) [cm]')
            axs[1].set_ylabel('est error: DLT error - geometric error [cm]')
            axs[1].grid()
            fig.savefig(os.path.join(output_dir, f'geometric_err_landmark{landmark_idx}.png'))
            plt.close()

            hist_range = (-0.2, 0.2)
            if len(values) > 0:
                print(landmark_idx, values.min(), values.max())
            else:
                print(landmark_idx, 'irrelevant')
            values[values>hist_range[1]] = hist_range[1]
            hist, bin_edges = np.histogram(values, bins=100, range=hist_range)
            dists.append(hist)
        dists = np.stack(dists)
        plt.plot(bin_edges[:-1], dists.transpose())
        plt.grid()
        plt.xlabel('[cm]')
        plt.title('diff = DLT error - geometric error')
        plt.legend(landmarks_legend)
        plt.savefig(os.path.join(output_dir, f'geometric_err_histogram.png'))
        plt.close()

    if False: # output 3d csv: DLTdv5_data_xyzpts.csv files
        landmarks_csv_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
        header = [f'pt{i + 1}_{a}' for i in range(17) for a in ['X', 'Y', 'Z']]
        for vid_name, outputs_list in model_output_3d.items():
            with open(f'outputs/{vid_name}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for l in outputs_list:
                    ll = np.zeros((17, 3)) * np.nan
                    ll[landmarks_csv_inds, :] = l
                    row = [str(i) for i in ll.flatten()]
                    row = ['NaN' if r == 'nan' else r for r in row]
                    writer.writerow(row)
            with open(f'outputs/{vid_name}_tal.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for l in labels_3d[vid_name]:
                    ll = np.zeros((17, 3)) * np.nan
                    ll[landmarks_csv_inds, :] = l
                    row = [str(i) for i in ll.flatten()]
                    row = ['NaN' if r == 'nan' else r for r in row]
                    writer.writerow(row)

    if False:  # plot diff histograms
        stacked_res = np.stack(res)

        print('Plot results')
        dists = []
        for landmark_idx in range(stacked_res.shape[1]):
            hist_range = (0, 10)
            values = stacked_res[~np.isnan(stacked_res[:, landmark_idx]), landmark_idx]
            values[values>hist_range[1]] = hist_range[1]
            hist, bin_edges = np.histogram(values, bins=100, range=hist_range)
            dists.append(hist)
        dists = np.stack(dists)
        plt.plot(bin_edges[:-1], dists.transpose())
        plt.grid()
        plt.xlabel('[cm]')
        plt.legend(landmarks_legend)
        plt.savefig(os.path.join(output_dir, f'dist_histogram.png'))
        plt.close()

    if True:  # plot characteristics
        print('Plot characteristics')
        # base vector: 8/9 -> 4
        # head angle: 0 -> 2
        # pectoral angle: 8/9 -> 6/7
        # tail angle: 5 -> 6
        final_params_dict = {}
        data_for_hist = []
        for vid_name, vid_data in model_output_3d.items():
            print(vid_name)
            char_mat = calc_characteristics(vid_data)
            # filtered_char = calc_local_mean_no_outliers(char_mat[:, 0], 2, 2)
            dlt_char_mat = calc_characteristics(model_output_3d_dlt[vid_name])
            label_char_mat = calc_characteristics(labels_3d[vid_name])

            _inds_dict = inds_dict.get(vid_name, [10, 20, 30])
            if True:
                for i_start in range(6):
                    for i_end in range(6):
                        _inds = [_inds_dict[0]+i_start, _inds_dict[1]-i_end, _inds_dict[2]]
                        label_final_params = FinalParams(_inds, label_char_mat, plate[vid_name][0],
                                                         np.stack(labels_3d[vid_name])[:, :2, :].mean(axis=1))
                        final_params = FinalParams(_inds, char_mat, plate[vid_name][0],
                                                   np.stack(vid_data)[:, :2, :].mean(axis=1))
                        data_for_hist.append([(label_final_params.gape_opening_speed[0] - final_params.gape_opening_speed[0]) / label_final_params.gape_opening_speed[0],
                                              (label_final_params.pect_speed_open[0] - final_params.pect_speed_open[0]) / label_final_params.pect_speed_open[0],
                                              (label_final_params.body_speed_open[0] - final_params.body_speed_open[0]) / label_final_params.body_speed_open[0],
                                              ])

            label_final_params = FinalParams(_inds_dict, label_char_mat, plate[vid_name][0],
                                             np.stack(labels_3d[vid_name])[:, :2, :].mean(axis=1))
            final_params = FinalParams(_inds_dict, char_mat, plate[vid_name][0],
                                       np.stack(vid_data)[:, :2, :].mean(axis=1))
            final_params_dict[vid_name] = [label_final_params, final_params]

            if False:  # plot mouth gape inds
                _is_gape = np.stack(is_gape[vid_name])
                _gape_inds = np.array(range(_is_gape.shape[0]))[_is_gape]
                fig = plt.figure(figsize=(15, 10))
                plt.plot(label_char_mat[:, 0], '.')
                plt.plot(_gape_inds, label_char_mat[_is_gape, 0], '.')
                # plt.plot(char_mat[:, 0], '.')
                # plt.plot(_gape_inds, char_mat[_is_gape, 0], '.')
                plt.xticks(np.arange(0, label_char_mat.shape[0], 2), rotation='vertical')
                plt.grid()
                fig.savefig(os.path.join(os.path.join(output_dir, f'bis_{vid_name}.png')))
                plt.close()

            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{vid_name} characteristics')
            axs[0, 0].plot(final_params.time_steps, char_mat[:, 0])
            if run_geometric:
                axs[0, 0].plot(final_params.time_steps, dlt_char_mat[:, 0], ':')
            axs[0, 0].plot(final_params.time_steps, label_char_mat[:, 0], '--')
            # axs[0, 0].plot(final_params.time_steps, filtered_char, '.-')
            label_final_params.plot_gape(axs[0, 0], 'k-', _inds_dict)
            final_params.plot_gape(axs[0, 0], 'k:', _inds_dict)

            # _min = min(filtered_char[~np.isnan(filtered_char)].min(), label_char_mat[~np.isnan(label_char_mat[:, 0]), 0].min())
            # _max = max(filtered_char[~np.isnan(filtered_char)].max(), label_char_mat[~np.isnan(label_char_mat[:, 0]), 0].max())
            _min = label_char_mat[~np.isnan(label_char_mat[:, 0]), 0].min()
            _max = label_char_mat[~np.isnan(label_char_mat[:, 0]), 0].max()
            axs[0, 0].set_ylim(bottom=_min, top=_max*1.2)
            axs[0, 0].set_title('Mouth Gape [cm]')
            if run_geometric:
                axs[0, 0].legend(['geometric', 'DLT', 'label', 'geometric filtered'])
            else:
                axs[0, 0].legend(['DLT', 'label', 'DLT filtered'])
            axs[0, 1].plot(final_params.time_steps, char_mat[:, 1])
            if run_geometric:
                axs[0, 1].plot(final_params.time_steps, dlt_char_mat[:, 1], ':')
            axs[0, 1].plot(final_params.time_steps, label_char_mat[:, 1], '--')
            label_final_params.plot_head(axs[0, 1], 'k-', _inds_dict)
            final_params.plot_head(axs[0, 1], 'k:', _inds_dict)
            axs[0, 1].set_title('Head Angle [rad]')

            axs[1, 0].plot(final_params.time_steps, char_mat[:, 2])
            if run_geometric:
                axs[1, 0].plot(final_params.time_steps, dlt_char_mat[:, 2])
            axs[1, 0].plot(final_params.time_steps, label_char_mat[:, 2], '--')
            label_final_params.plot_pect(axs[1, 0], 'k-', _inds_dict)
            final_params.plot_pect(axs[1, 0], 'k:', _inds_dict)
            axs[1, 0].set_title('Pectoral Angle [rad]')

            axs[1, 1].plot(final_params.time_steps, char_mat[:, 3])
            if run_geometric:
                axs[1, 1].plot(final_params.time_steps, dlt_char_mat[:, 3])
            axs[1, 1].plot(final_params.time_steps, label_char_mat[:, 3], '--')
            label_final_params.plot_tail(axs[1, 1], 'k-', _inds_dict)
            final_params.plot_tail(axs[1, 1], 'k:', _inds_dict)
            axs[1, 1].set_title('Tail Angle [rad]')

            for ax in axs.flat:
                if ax.legend_ is None:
                    ax.legend(['geometric', 'DLT', 'label'])
                ax.grid()

            fig.savefig(os.path.join(os.path.join(output_dir, f'char_{vid_name}.png')))
            plt.close()

        # plot sampling effect
        hist_range = [-4, 4]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'effect of sampling "open" indices: (label-output)/label')

        _data_for_hist = np.stack(data_for_hist)[:, 0]
        hist, bin_edges = np.histogram(_data_for_hist, bins=100, range=hist_range)
        axs[0, 0].bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=bin_edges[1] - bin_edges[0])
        axs[0, 0].set_ylabel('gape_opening_speed diff')
        axs[0, 0].grid()
        _data_for_hist = np.stack(data_for_hist)[:, 1]
        hist, bin_edges = np.histogram(_data_for_hist, bins=100, range=hist_range)
        axs[0, 1].bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=bin_edges[1] - bin_edges[0])
        axs[0, 1].set_ylabel('pect_speed_open diff')
        axs[0, 1].grid()
        _data_for_hist = np.stack(data_for_hist)[:, 2]
        hist, bin_edges = np.histogram(_data_for_hist, bins=100, range=hist_range)
        axs[1, 0].bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=bin_edges[1] - bin_edges[0])
        axs[1, 0].set_ylabel('body_speed_open diff')
        axs[1, 0].grid()
        _data_for_hist = np.stack(data_for_hist).flatten()
        hist, bin_edges = np.histogram(_data_for_hist, bins=100, range=hist_range)
        axs[1, 1].bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=bin_edges[1] - bin_edges[0])
        axs[1, 1].set_ylabel('all diff')
        axs[1, 1].grid()

        fig.savefig(os.path.join(os.path.join(output_dir, f'mouth_open_sampling_effect.png')))
        plt.close()

        # plot final params vs. label for all videos
        fp = np.zeros((len(final_params.final_params_names), len(model_output_3d), 2))
        for vid_index, vid_name in enumerate(model_output_3d):
            for j, fp_name in enumerate(final_params.final_params_names):
                label_param = final_params_dict[vid_name][0].__getattribute__(fp_name)[0]
                measured_param = final_params_dict[vid_name][1].__getattribute__(fp_name)[0]
                fp[j, vid_index, :] = np.array([label_param, measured_param])

        markers = ["o", "v", "^", "<" ,">" ,"1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x"]

        fig, axs = plt.subplots(4, 4, figsize=(15, 10))
        fig.suptitle(f'final params vs. labels')
        for j, fp_name in enumerate(final_params.final_params_names):
            ax = axs[int(j / 4), j % 4]
            for i_marker, (_l, _m) in enumerate(zip(fp[j, :, 0], fp[j, :, 1])):
                ax.plot(_l, _m, markers[i_marker])
            _label = fp[j, ~np.isnan(fp[j, :, 0]), 0]
            ax.plot([_label.min(), _label.max()],
                    [_label.min(), _label.max()], 'k--')
            _data = fp[j, ~np.isnan(fp[j, :, 0]), 1]
            a, b = calc_ls(_label, _data)
            ax.plot([_label.min(), _label.max()],
                    [a*_label.min()+b, a*_label.max()+b], 'r--')
            a, b = calc_ransac(_label, _data)
            ax.plot([_label.min(), _label.max()],
                    [a*_label.min()+b, a*_label.max()+b], 'c:')
            ax.set_title(fp_name)
            ax.grid()

        for i_marker, (_l, _m) in enumerate(zip(fp[:, :, 0].transpose(), fp[:, :, 1].transpose())):
            axs[3, 3].plot(_l, _m, markers[i_marker])
        _label = fp[..., 0].flatten()
        _data = fp[..., 1].flatten()[~np.isnan(_label)]
        _label = _label[~np.isnan(_label)]
        axs[3, 3].plot([_label.min(), _label.max()],
                       [_label.min(), _label.max()], 'k--')
        a, b = calc_ls(_label, _data)
        axs[3, 3].plot([_label.min(), _label.max()],
                       [a * _label.min() + b, a * _label.max() + b], 'r--')
        a, b = calc_ransac(_label, _data, iters=50)
        axs[3, 3].plot([_label.min(), _label.max()],
                [a * _label.min() + b, a * _label.max() + b], 'c:')
        axs[3, 3].set_title('all')
        axs[3, 3].grid()

        for i_marker, _m in enumerate(range(fp.shape[1])):
            axs[3, 2].plot(0, _m, markers[i_marker])
        axs[3, 2].legend(sorted(model_output_3d.keys()))
        fig.savefig(os.path.join(output_dir, f'final_params_vs_label.png'))
        plt.close()

        with open(os.path.join(output_dir, 'report_final_params.pickle'), 'wb') as f:
            pickle.dump(final_params_dict, f)

