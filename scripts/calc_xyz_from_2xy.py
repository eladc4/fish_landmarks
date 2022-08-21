from os.path import join
import numpy as np
import csv
import cv2
from PIL import Image, ImageDraw

num_keypoints = 17
# relevant_keypoints = np.arange(num_keypoints)
relevant_keypoints = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])


def read_cal_matrix(cal_file):
    with open(cal_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        cal_mat = np.zeros((12, 2))
        for i, row in enumerate(csv_reader):
            cal_mat[i, :] = np.array(row)

        # reshape & transpose
        cal_mat[:, 0].reshape((3, 4))

        r = cal_mat[:, 0].reshape((3, 4))[:, :-1]
        print(np.matmul(r.transpose(), r))
        print(np.matmul(r, r.transpose()))

    return cal_mat


def read_labels(label_file, vid_height):
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


def read_3d_labels(label_file, vid_height):
    with open(label_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        keypoints_3d = []
        for row in csv_reader:
            if len(row) > 3*num_keypoints:
                raise Exception('More than 17 keypoints')
            # need to flip y-axis, as axes in matlab are reversed compared to python
            # pt1_X	pt1_Y	pt1_Z
            keypoints_3d.append(np.array([(float(row.get(f'pt{i}_X', float('nan'))),
                                           vid_height - float(row.get(f'pt{i}_Y', float('nan'))),
                                           float(row.get(f'pt{i}_Z', float('nan'))))
                                          for i in range(1, num_keypoints + 1)])[relevant_keypoints, :])
    return keypoints_3d


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


base_path="/data/projects/swat/users/eladc/project/dataset_src/all_dig_data"
vid1_csv = join(base_path, "10.05.19", "1", "dig", "cut_dig", "1", "f_20190510_150806_20190510_151419.avi")
vid2_csv = join(base_path, "10.05.19", "1", "dig", "cut_dig", "1", "r_20190510_150806_20190510_151832.avi")
landmarks_2d_csv = join(base_path, "10.05.19", "1", "dig", "cut_dig", "1", "DLTdv5_data_xypts.csv")
landmarks_3d_csv = join(base_path, "10.05.19", "1", "dig", "cut_dig", "1", "DLTdv5_data_xyzpts.csv")
coeffs_csv = join(base_path, "10.05.19", "cal", "cal01_DLTcoefs.csv")

cams_vidcap = [cv2.VideoCapture(vid1_csv),
               cv2.VideoCapture(vid2_csv)]
cams_kpts = read_labels(landmarks_2d_csv,
                        cams_vidcap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
kpts_3d = read_3d_labels(landmarks_3d_csv,
                              cams_vidcap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
mat1, mat2 = read_cal_matrix(coeffs_csv)

img_pairs = [[], []]
for cam_index, cam_vidcap in enumerate(cams_vidcap):
    cam_name = f'cam{cam_index}'
    _img_idx = 0
    while True:
        success, image = cam_vidcap.read()
        if not success:
            break
        img = Image.fromarray(image)
        label = cams_kpts[cam_index][_img_idx].copy()
        # img = show_image_with_keypoints(img, label)
        _img_idx += 1
        img_pairs[cam_index].append(img)




