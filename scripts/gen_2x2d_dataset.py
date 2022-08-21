import pickle
import os
import csv
import cv2
from PIL import Image, ImageDraw
import numpy as np

from scripts.videos_and_csv_labels import get_videos_and_csv_labels_list


def read_cal_matrix(cal_file):
    with open(cal_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        cal_mat = np.ones((12, 2))
        for i, row in enumerate(csv_reader):
            cal_mat[i, :] = np.array(row)
    return cal_mat


def read_labels(label_file, vid_height):
    num_keypoints = 17
    relevant_keypoints = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 6])  # keypoint 14 is only for marking gape time, 6 for marking the plate
    with open(label_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        cam1_keypoints, cam2_keypoints = [], []
        for row in csv_reader:
            if len(row) > 4*num_keypoints:
                raise Exception('More than 17 keypoints')
            # need to flip y-axis, as axes in matlab are reversed compared to python
            cam1_keypoints.append(np.array([(float(row.get(f'pt{i}_cam1_X', float('nan'))),
                                             vid_height-float(row.get(f'pt{i}_cam1_Y', float('nan'))))
                                            for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
            cam2_keypoints.append(np.array([(float(row.get(f'pt{i}_cam2_X', float('nan'))),
                                             vid_height-float(row.get(f'pt{i}_cam2_Y', float('nan'))))
                                            for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
    return cam1_keypoints, cam2_keypoints


def show_image_with_keypoints(i, keypoints):
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    if not isinstance(i, Image.Image):
        i = Image.fromarray(i)
    source_img = i.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for _, k in enumerate(keypoints):
        if not np.any(np.isnan(k)):
            # k[1] = i.size[1]-k[1]
            if _< 10:
                draw.rectangle((tuple(k-2), tuple(k+2)), outline="cyan")
                if keypoints.shape[0] > 10:
                    draw.line((tuple(k), tuple(keypoints[_+10])), fill='yellow')
            else:
                draw.rectangle((tuple(k-2), tuple(k+2)), outline="red")
                draw.text(tuple(k[:2]), str(_-10), fill='white')
    # source_img.convert('RGB').show()
    return source_img.convert('RGB')


if __name__ == '__main__':
    dataset_folder = '/data/projects/swat/users/eladc/project/dataset_2x2d'
    with_labels = False
    labels = []
    is_gape_list = []
    plate_list = []
    cal_mats = []
    img_index = 0
    videos_and_csv_labels_list = get_videos_and_csv_labels_list()
    for i, vid_and_label in enumerate(videos_and_csv_labels_list):
        vid_name = f'vid_{i:03d}'
        cams_vidcap = [cv2.VideoCapture(vid_and_label.video1),
                       cv2.VideoCapture(vid_and_label.video2)]
        cams_kpts = read_labels(vid_and_label.label_csv,
                                cams_vidcap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        cal_matrix = read_cal_matrix(vid_and_label.cal_csv)
        cal_mats.append(cal_matrix)

        _labels = [[], []]
        _is_gape = [[], []]
        _plate = [[], []]
        for cam_index, cam_vidcap in enumerate(cams_vidcap):
            cam_name = f'cam{cam_index}'
            _img_idx = 0
            while True:
                success, image = cam_vidcap.read()
                if not success:
                    break
                img = Image.fromarray(image)
                label = cams_kpts[cam_index][_img_idx][:10, :].copy()
                is_gape = ~np.isnan(cams_kpts[cam_index][_img_idx][10, 0].copy())
                plate = cams_kpts[cam_index][_img_idx][11, :].copy()
                if with_labels:
                    img = show_image_with_keypoints(img, label)
                output_img_path = os.path.join(dataset_folder, vid_name, cam_name)
                os.makedirs(output_img_path, exist_ok=True)
                img.save(os.path.join(output_img_path, f'img_{img_index:06}.jpg'))
                _labels[cam_index].append(label)
                _is_gape[cam_index].append(is_gape)
                _plate[cam_index].append(plate)
                _img_idx += 1
                img_index += 1

            # if cam_index==0:
            #     print(output_img_path.split('/')[-2] ,'-', os.path.dirname(vid_and_label.video1).split('dataset_src/')[-1])
            print(f"{i+1}.{cam_index+1}/{len(videos_and_csv_labels_list)}: Finished processing {_img_idx}"
                  f" images from {os.path.dirname(vid_and_label.video1)}"
                  f" ({cam_index+1}/{len(cams_vidcap)})")
        labels.append(_labels)
        is_gape_list.append(_is_gape)
        plate_list.append(_plate)
    print(img_index)

    if not with_labels:
        with open(os.path.join(dataset_folder, f'labels.pickle'), 'wb') as f:
            pickle.dump(labels, f)

        with open(os.path.join(dataset_folder, f'gape.pickle'), 'wb') as f:
            pickle.dump(is_gape_list, f)

        with open(os.path.join(dataset_folder, f'plate.pickle'), 'wb') as f:
            pickle.dump(plate_list, f)

        with open(os.path.join(dataset_folder, f'cal_mats.pickle'), 'wb') as f:
            pickle.dump(cal_mats, f)
