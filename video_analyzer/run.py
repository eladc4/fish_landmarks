import os
import csv
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from matplotlib.backend_bases import MouseButton
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt

from scripts.gen_2x2d_dataset import read_cal_matrix
from datasets.dataset_2d import show_image_with_keypoints
from model_evaluation.epipolar_geometry_utils import calc_3d_from_2x2d

from lib.config import cfg
import lib.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Run video analyzer')
    # videos
    parser.add_argument('--vid1',
                        help='video from camera #1',
                        type=str, required=True)
    parser.add_argument('--vid2',
                        help='video from camera #2',
                        type=str, required=True)

    # calibration matrix
    parser.add_argument('--cal',
                        help='cameras calibration matrix',
                        type=str, default=None, required=False)

    # model config (trained parameters specified inside)
    parser.add_argument('--cfg',
                        help='model config',
                        type=str, required=True)

    # output folder to save digitized CSVs
    parser.add_argument('--output_folder',
                        help='output folder',
                        type=str, required=False, default=None)

    # Start index of frame in video
    parser.add_argument('--start',
                        help='Start index of frame in video',
                        type=int, required=False, default=None)

    # End index of frame in video
    parser.add_argument('--end',
                        help='End index of frame in video',
                        type=int, required=False, default=None)

    # A flag whether to save images with detected labels to chip
    parser.add_argument('--print_images',
                        help='Save images with detected labels',
                        required=False, action='store_true', default=False)

    _args = parser.parse_args()
    return _args


class VidCrop:
    def __init__(self, x0, y0, x1, y1, scale_x, scale_y):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.scale_x = scale_x
        self.scale_y = scale_y

    @property
    def scale_array(self):
        return np.array([self.scale_x, self.scale_y])

    @property
    def crop_array(self):
        return np.array([self.x0, self.y0])

    def __repr__(self):
        return f'p0 = ({self.x0}, {self.y0}), p1 = ({self.x1}, {self.y1}), scale=({self.scale_x}, {self.scale_y})'


def read_images_from_video(_vid):
    vidcap = cv2.VideoCapture(_vid)
    _imgs = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        _imgs.append(image.mean(-1).astype(np.uint8))
    return _imgs


def get_crop_inds(_img):
    def onselect(eclick, erelease):
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        ax.set_ylim(erelease.ydata, eclick.ydata)
        ax.set_xlim(eclick.xdata, erelease.xdata)
        fig.canvas.draw()

    fig = plt.figure(figsize=[7, 9])
    ax = fig.add_subplot(111)
    plt.imshow(_img)
    rs = widgets.RectangleSelector(
        ax, onselect, drawtype='box',
        rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
    plt.show()

    crop_inds = ((round(rs.ax.get_ylim()[1]), round(rs.ax.get_ylim()[0])),
                 (round(rs.ax.get_xlim()[0]), round(rs.ax.get_xlim()[1])))
    return crop_inds


def crop_video(_vid_imgs, input_size, inds=None):
    _img = np.stack([_vid_imgs[0], _vid_imgs[int(len(_vid_imgs)/2)], _vid_imgs[-1]], axis=2)
    if inds:
        ((x0, y0), (x1, y1)) = inds
    else:
        ((y0, y1), (x0, x1)) = get_crop_inds(_img)
    if y1-y0 > x1-x0:
        offset = round(((y1 -y0) - (x1 - x0)) / 2)
        x0 = x0 - offset
        x1 = x0 + (y1 - y0)
    else:
        offset = round(((x1-x0) - (y1-y0))/2)
        y0 = y0 - offset
        y1 = y0 + (x1-x0)

    scale_y, scale_x = (y1-y0)/input_size[0], (x1-x0)/input_size[1]
    out_imgs = []
    for i in range(len(_vid_imgs)):
        # consider cropping with PIL according to VAL_CROP_RATIO
        out_imgs.append(Image.fromarray(_vid_imgs[i]).crop((x0, y0, x1, y1)).resize(input_size))

    return out_imgs, VidCrop(x0, y0, x1, y1, scale_x, scale_y)


def run_model(model, pil_img, use_cuda):
    img = (np.array(pil_img, dtype=np.float32) - 127.5) / 127.5
    img = torch.Tensor(img[np.newaxis, np.newaxis, ...])
    if use_cuda:
        img = img.cuda()
    out = model(img)
    if use_cuda:
        out = out.cpu()
    out = out.detach().numpy()[0, ...]
    return out


if __name__ == '__main__':
    args = parse_args()

    output_folder = os.path.dirname(args.vid1) if args.output_folder is None else args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # load config
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # run model on videos
    cal_matrix = read_cal_matrix(args.cal) if args.cal is not None else False
    vid1_imgs = read_images_from_video(args.vid1)
    vid2_imgs = read_images_from_video(args.vid2)
    if len(vid1_imgs) != len(vid2_imgs):
        raise Exception('Error: Videos not the same length')
    if vid1_imgs[0].shape != vid2_imgs[0].shape:
        raise Exception('Error: Videos not the same length')
    H, W = vid1_imgs[0].shape
    _start = 0 if args.start is None else args.start
    _end = len(vid1_imgs) if args.end is None else args.end
    vid1_imgs = vid1_imgs[_start:_end]
    vid2_imgs = vid2_imgs[_start:_end]
    vid_imgs_list = [vid1_imgs, vid2_imgs]
    num_images = len(vid1_imgs)
    print('Loaded videos and calibration coefficients')

    vid1_cropped_imgs, vid1_crop = crop_video(vid1_imgs, cfg.MODEL.IMAGE_SIZE)
    print(f'Cropping video #1 according to points: {vid1_crop}')
    vid2_cropped_imgs, vid2_crop = crop_video(vid2_imgs, cfg.MODEL.IMAGE_SIZE)
    print(f'Cropping video #2 according to points: {vid2_crop}')
    vid_crops = [vid1_crop, vid2_crop]

    use_cuda = torch.cuda.is_available()
    # load a trained model
    if cfg.MODEL.NAME.lower() == 'lpn':
        get_model = models.lpn.get_pose_net
    elif cfg.MODEL.NAME.lower() == 'pose_hrnet':
        get_model = models.pose_hrnet.get_pose_net
    else:
        raise NotImplemented
    model = get_model(cfg, True)
    if use_cuda:
        model = model.cuda()
    print('Model loaded')

    th = 0.15
    adjust = True
    model_scale = np.array(cfg.MODEL.IMAGE_SIZE) / np.array(cfg.MODEL.HEATMAP_SIZE)
    num_landmarks = cfg.MODEL.NUM_JOINTS
    labels = [[], []]
    for i_vid, _vid_imgs in enumerate([vid1_cropped_imgs, vid2_cropped_imgs]):
        print(f'\nAnalyzing video #{i_vid}')
        for i in tqdm(range(num_images)):
            out = run_model(model, _vid_imgs[i], use_cuda)

            max_inds = np.argmax(out.reshape((num_landmarks, -1)), axis=1)
            y, x = np.floor(max_inds/cfg.MODEL.HEATMAP_SIZE[1]).astype(np.int64), max_inds%cfg.MODEL.HEATMAP_SIZE[1]
            _labels = np.stack([x, y], 1).astype(np.float32)

            # remove labels beneath threshold
            _labels[out[np.arange(10), y, x] < th, :] = np.nan

            if adjust:
                pixel_th = 0.05
                pixel_shift = 0.25
                x_offset = np.zeros(num_landmarks)
                max_y, max_x = out.shape[1:]
                diff = out[np.arange(num_landmarks), y, np.minimum(x+1, max_x-1)] - out[np.arange(num_landmarks), y, np.maximum(x-1, 0)]
                x_offset[diff < -pixel_th] = -pixel_shift
                x_offset[diff > pixel_th] = pixel_shift
                y_offset = np.zeros(num_landmarks)
                diff = out[np.arange(num_landmarks), np.minimum(y+1, max_y-1), x] - out[np.arange(num_landmarks), np.maximum(y-1, 0), x]
                y_offset[diff < -pixel_th] = -pixel_shift
                y_offset[diff > pixel_th] = pixel_shift

                _labels += np.stack([x_offset, y_offset], 1)

            _labels = _labels * model_scale
            # show_image_with_keypoints(_vid_imgs[i], _labels).show()

            # move labels to original image coordinates
            _labels = _labels * vid_crops[i_vid].scale_array + vid_crops[i_vid].crop_array
            # show_image_with_keypoints(Image.fromarray(Image.fromarray(vid_imgs_list[i_vid][i])), _labels).show()
            if args.print_images:
                os.makedirs(os.path.join(output_folder, 'imgs'), exist_ok=True)
                show_image_with_keypoints(Image.fromarray(vid_imgs_list[i_vid][i]),
                                          _labels).save(os.path.join(output_folder, 'imgs', f'{i_vid}_{i:04d}.jpg'))

            _labels[:, 1] = H - _labels[:, 1]
            labels[i_vid].append(_labels)

    out_3d = []
    if cal_matrix:
        print(f'\nAnalyzing 3D points')
        for i in tqdm(range(num_images)):
            out_3d.append(calc_3d_from_2x2d(labels[0][i], labels[1][i], cal_matrix, geometric=False)[0])

    landmarks_csv_inds = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    total_landmarks = 15

    _file = os.path.join(output_folder, 'xypts.csv')
    header = [f'pt{i + 1}_{_cam}_{a}' for i in range(total_landmarks) for _cam in ['cam1', 'cam2'] for a in ['X', 'Y']]
    with open(_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for l1, l2 in zip(labels[0], labels[1]):
            ll = np.zeros((total_landmarks, 4)) * np.nan
            ll[landmarks_csv_inds, :] = np.concatenate([l1, l2], 1)
            row = [str(float(i)+0.01) for i in ll.flatten()]
            row = ['NaN' if r == 'nan' else r for r in row]
            writer.writerow(row)
    print('Finished writing xy.csv:', _file)

    if out_3d:
        # Write xyzpts.csv
        _file = os.path.join(output_folder, 'xyzpts.csv')
        header = [f'pt{i + 1}_{a}' for i in range(total_landmarks) for a in ['X', 'Y', 'Z']]
        with open(_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for landmarks_3d in out_3d:
                ll = np.zeros((total_landmarks, 3)) * np.nan
                ll[landmarks_csv_inds, :] = landmarks_3d
                row = [str(float(i)) for i in ll.flatten()]
                row = ['NaN' if r == 'nan' else r for r in row]
                writer.writerow(row)
        print('Finished writing xyz.csv:', _file)

        # Write dummy xyzres.csv
        _file = os.path.join(output_folder, 'xyzres.csv')
        header = [f'pt{i + 1}_dltres' for i in range(total_landmarks)]
        with open(_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(labels[0])):
                writer.writerow([0.001] * len(header))
        print('Finished writing dummy xyzres.csv:', _file)

        # Write dummy offsets.csv
        _file = os.path.join(output_folder, 'offsets.csv')
        header = ['cam1_offset', 'cam2_offset']
        with open(_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(labels[0])):
                writer.writerow([0, 0])
        print('Finished writing dummy offsets.csv:', _file)
    else:
        print('Skipping xyz.csv, xyzres.csv & offsets.csv as no calibration matrix is missing')



