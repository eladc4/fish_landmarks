import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets.transforms as np_transforms
from datasets.dataset_2d import Dataset2D
from datasets.dataset_2d import show_image_with_keypoints
import pickle
import csv
from PIL import Image


output_dir = '/fish_landmarks/dataset_analysis'

hist_size = (-400, 400)
prob_map_size = (1000, 1000)
train_epochs = 1
save_images = True


def get_hist(points_list):
    _points_list = np.array(points_list)
    # min_x, min_y = train_lip_to_back.min(axis=0)
    # max_x, max_y = train_lip_to_back.max(axis=0)
    # fig = plt.figure()
    # plt.scatter(points_list[:, 0], points_list[:, 1])

    H, xedges, yedges = np.histogram2d(_points_list[:, 0], _points_list[:, 1],
                                       bins=(np.linspace(*hist_size, num=50),
                                             np.linspace(*hist_size, num=50)))
    return cv2.resize(H.T, prob_map_size)


def iterate_dataset(_train_epochs, _data_loader, get_landmark_stats=False, outpath='train'):
    if save_images:
        os.makedirs(os.path.join(output_dir, outpath), exist_ok=True)
    mouth_vector = []
    head_vector = []
    pectoral_vector = []
    tail_vector = []
    dists = []
    diff_depth = 5
    vid_cam_name = None
    labels_per_img = []
    img_idx = 0
    for epoch in range(_train_epochs):
        for i, (data, target, target_weight, pp) in enumerate(_data_loader):
            print(f'epoch {epoch+1}/{_train_epochs} {i+1}/{len(_data_loader)}')
            for j in range(data.size(0)):
                _label = target.cpu().detach().numpy()

                if save_images:
                    img = Image.fromarray((data[j, 0, ...].cpu().detach().numpy()*127.5+127.5).astype(np.uint8))
                    show_image_with_keypoints(img, _label[j, ...]).save(os.path.join(output_dir, outpath, f'img_{img_idx:05d}.jpg'))
                    img_idx += 1

                if get_landmark_stats:
                    _vid_cam_name = pp[0][j].split(os.sep)[-2]
                    if vid_cam_name == _vid_cam_name:
                        labels_per_img.append(_label[j, ...])
                    else:
                        if labels_per_img:
                            labels_per_img = np.stack(labels_per_img)
                            _dists = []
                            for diff_idx in range(diff_depth):
                                _diff = labels_per_img[diff_idx+1:] - labels_per_img[0:-diff_idx-1]
                                _dists.append(np.linalg.norm(_diff, axis=2))
                            dists.append(_dists)
                        labels_per_img = [_label[j, ...]]
                    vid_cam_name = _vid_cam_name

                offset = 0 if np.isnan(_label[j, 7, 0]) else 1
                if not np.isnan(_label[j, 0, 0]) and not np.isnan(_label[j, 1, 0]):
                    mouth_vector.append(_label[j, 0, :] - _label[j, 1, :])
                if not np.isnan(_label[j, 0, 0]) and not np.isnan(_label[j, 2, 0]):
                    head_vector.append(_label[j, 2, :] - _label[j, 0, :])
                if not np.isnan(_label[j, 6 + offset, 0]) and not np.isnan(_label[j, 8 + offset, 0]):
                    pectoral_vector.append(_label[j, 6 + offset, :] - _label[j, 8 + offset, :])
                if not np.isnan(_label[j, 5, 0]) and not np.isnan(_label[j, 4, 0]):
                    tail_vector.append(_label[j, 5, :] - _label[j, 4, :])

    landmark_stats = np.zeros((10, 2, diff_depth))
    if get_landmark_stats:
        for diff_idx in range(diff_depth):
            dists_array = np.concatenate([_dists[diff_idx] for _dists in dists], axis=0)
            for landmark_idx in range(dists_array.shape[1]):
                d = dists_array[~np.isnan(dists_array[:, landmark_idx]), landmark_idx]
                landmark_stats[landmark_idx, 0, diff_idx] = d.mean()
                landmark_stats[landmark_idx, 1, diff_idx] = d.std()

    return get_hist(mouth_vector), get_hist(head_vector), get_hist(pectoral_vector), get_hist(tail_vector), landmark_stats


# analyze dataset distribution
# 0: top lip
# 1: bottom lip
# 2: front top fin
# 3: middle top fin
# 4: back top fin
# 5: top back fin
# 6: left fin - edge
# 7: right fin - edge
# 8: left fin - base
# 9: right fin - base

VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]
train_dataset = Dataset2D(train=True, val_split=VAL_SPLIT, reutrn_target_as_heatmaps=False, num_input_images=1,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(127.5, 127.5)]),
                          np_transforms=[np_transforms.FlipLeftRight(),
                                         np_transforms.CropLandmarks(ratio=3.0),
                                         np_transforms.RandomRotateScale(max_angle=40, scale_limits=[0.7, 2.0]),
                                         np_transforms.RandomSquash(40),
                                         np_transforms.CropLandmarks(ratio=1.5),
                                         np_transforms.Resize(image_size=(384, 384))],
                          )
valid_dataset = Dataset2D(train=False, val_split=VAL_SPLIT, reutrn_target_as_heatmaps=False, num_input_images=1,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(127.5, 127.5)]),
                          np_transforms=[np_transforms.CropLandmarks(ratio=2.0),
                                         np_transforms.Resize(image_size=(384, 384)),
                                         ],
                          )

# VAL_SPLIT = list(range(41))
# valid_dataset = Dataset2D(train=False, val_split=VAL_SPLIT, reutrn_target_as_heatmaps=False,
#                           transform=transforms.Compose([transforms.ToTensor()]),
#                           )

num_workers = 0
valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)

hists = iterate_dataset(1, valid_loader, get_landmark_stats=True, outpath='valid')
valid_mouth_vector_hist, valid_head_vector_hist, valid_pectoral_vector_hist, valid_tail_vector_hist, diff_probs = hists

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
)

hists = iterate_dataset(train_epochs, train_loader, outpath='train')
train_mouth_vector_hist, train_head_vector_hist, train_pectoral_vector_hist, train_tail_vector_hist, _ = hists

img_mouth_vector = Image.fromarray(np.stack([np.uint8(train_mouth_vector_hist / train_mouth_vector_hist.max() * 255 * 10),
                                             np.uint8(valid_mouth_vector_hist / valid_mouth_vector_hist.max() * 255 * 10),
                                             np.zeros(prob_map_size, dtype=np.uint8)], 2))

img_head_vector = Image.fromarray(np.stack([np.uint8(train_head_vector_hist / train_head_vector_hist.max() * 255 * 10),
                                            np.uint8(valid_head_vector_hist / valid_head_vector_hist.max() * 255 * 10),
                                            np.zeros(prob_map_size, dtype=np.uint8)], 2))

img_pectoral_vector = Image.fromarray(np.stack([np.uint8(train_pectoral_vector_hist / train_pectoral_vector_hist.max() * 255 * 10),
                                                np.uint8(valid_pectoral_vector_hist / valid_pectoral_vector_hist.max() * 255 * 10),
                                                np.zeros(prob_map_size, dtype=np.uint8)], 2))

img_tail_vector = Image.fromarray(np.stack([np.uint8(train_tail_vector_hist / train_tail_vector_hist.max() * 255 * 10),
                                            np.uint8(valid_tail_vector_hist / valid_tail_vector_hist.max() * 255 * 10),
                                            np.zeros(prob_map_size, dtype=np.uint8)], 2))

if False:
    img_mouth_vector.show(title='Mouth Vector')
    img_head_vector.show(title='Head Vector')
    img_pectoral_vector.show(title='Pectoral Vector')
    img_tail_vector.show(title='Tail Vector')

img_mouth_vector.save(os.path.join(output_dir, 'mouth_vector.png'))
img_head_vector.save(os.path.join(output_dir, 'head_vector.png'))
img_pectoral_vector.save(os.path.join(output_dir, 'pectoral_vector.png'))
img_tail_vector.save(os.path.join(output_dir, 'tail_vector.png'))

header = []
for diff_idx in range(diff_probs.shape[2]):
    header.append(f'diff {diff_idx+1}')
    header.append('MEAN')
    header.append('STD')
    header.append('')

with open(os.path.join(output_dir, 'diff_stats.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(10):
        row = []
        for diff_idx in range(diff_probs.shape[2]):
            row.append(f'landmark {i}')
            row.append(np.around(diff_probs[i, 0, diff_idx], decimals=2))
            row.append(np.around(diff_probs[i, 1, diff_idx], decimals=2))
            row.append('')
        writer.writerow(row)

with open(os.path.join(output_dir, 'diff_stats.pickle'), 'wb') as f:
    pickle.dump(diff_probs, f)

print('Done')