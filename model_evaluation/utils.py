
import csv

import numpy as np
import cv2


def calc_ransac(x: np.ndarray, y: np.ndarray, iters=40, eps=None):
    if eps is None:
        eps = (x.max() - x.min())/10

    x = np.stack([x, np.ones(x.shape)]).transpose()

    res_dict = {}
    for i in range(iters):
        i0 = np.random.randint(low=0, high=x.shape[0])
        i1 = np.random.randint(low=0, high=x.shape[0])
        while i0 == i1:
            i1 = np.random.randint(low=0, high=x.shape[0])

        _inds = np.array([i0, i1])
        a = np.linalg.inv(x[_inds, :].transpose() @ x[_inds, :]) @ x[_inds, :].transpose() @ y[_inds]
        _a, _b = a
        x_ = (_a*(y-_b)+x[:, 0]) / (1+_a**2)
        y_ = _a*x_+_b
        res_dict[f'{i0}_{i1}'] = [np.sum(np.abs(_a*x[:,0]+_b-y) < eps), a]

    max_a = 0
    best_a = None
    for k, v in res_dict.items():
        if v[0] > max_a:
            max_a = v[0]
            best_a = v[1]

    inlier_inds = np.abs(x@best_a-y) < eps
    a = np.linalg.inv(x[inlier_inds, :].transpose() @ x[inlier_inds, :]) @ x[inlier_inds, :].transpose() @ y[inlier_inds]
    return a


def calc_ls(_A, _b):
    if len(_A.shape) == 1:
        _A = np.stack([_A, np.ones(_A.shape)]).transpose()
    try:
        a = np.linalg.lstsq(_A, _b, rcond=None)[0]
    except:
        a = np.linalg.inv(_A.transpose() @ _A) @ _A.transpose() @ _b
    return a


class FinalParams:
    final_params_names = ['gape_opening_speed', 'gape_closing_speed', 'gape_flick_speed',
                          'head_body_angular_speed_close', 'head_body_angular_speed_flick',
                          'tail_body_angular_speed_close', 'tail_body_angular_speed_flick',
                          'pect_speed_open', 'pect_speed_close', 'pect_speed_flick',
                          'body_speed_open', 'body_speed_close', 'body_speed_flick']

    def __init__(self, bis_inds, char_mat, plate_3d, mouth_center_3d):
        self.flick_samples = 8
        self.frame_rate = 200
        X_open = np.arange(bis_inds[0], bis_inds[1])
        X_close = np.arange(bis_inds[1], bis_inds[2])
        X_flick = np.arange(bis_inds[2], bis_inds[2] + self.flick_samples)
        time_steps = np.arange(char_mat.shape[0]) / self.frame_rate
        dist = np.linalg.norm(mouth_center_3d - plate_3d, axis=1)
        self.time_steps = time_steps
        self.dist = dist

        self.gape_opening_speed = calc_ls(time_steps[X_open], char_mat[X_open, 0])
        self.gape_closing_speed = calc_ls(time_steps[X_close], char_mat[X_close, 0])
        self.gape_flick_speed = calc_ls(time_steps[X_flick], char_mat[X_flick, 0])
        self.head_body_angular_speed_close = calc_ls(time_steps[X_close], char_mat[X_close, 1])
        self.tail_body_angular_speed_close = calc_ls(time_steps[X_close], char_mat[X_close, 3])
        self.head_body_angular_speed_flick = calc_ls(time_steps[X_flick], char_mat[X_flick, 1])
        self.tail_body_angular_speed_flick = calc_ls(time_steps[X_flick], char_mat[X_flick, 3])
        self.pect_speed_open = calc_ls(time_steps[X_open], char_mat[X_open, 2])
        self.pect_speed_close = calc_ls(time_steps[X_close], char_mat[X_close, 2])
        self.pect_speed_flick = calc_ls(time_steps[X_flick], char_mat[X_flick, 2])
        self.body_speed_open = calc_ls(time_steps[X_open], dist[X_open])
        self.body_speed_close = calc_ls(time_steps[X_close], dist[X_close])
        self.body_speed_flick = calc_ls(time_steps[X_flick], dist[X_flick])

    def _plot(self, ax, line_str, x, line_params):
        y = x * line_params[0] + line_params[1]
        ax.plot(x, y, line_str)

    def plot_gape(self, ax, line_str, inds_dict):
        # open
        self._plot(ax, line_str, self.time_steps[inds_dict[:2]], self.gape_opening_speed)
        # close
        self._plot(ax, line_str, self.time_steps[inds_dict[1:]], self.gape_closing_speed)
        # flick
        self._plot(ax, line_str, self.time_steps[inds_dict[-1] + np.array([0, self.flick_samples])], self.gape_flick_speed)

    def plot_head(self, ax, line_str, inds_dict):
        # close
        self._plot(ax, line_str, self.time_steps[inds_dict[1:]], self.head_body_angular_speed_close)
        # flick
        self._plot(ax, line_str, self.time_steps[inds_dict[-1] + np.array([0, self.flick_samples])], self.head_body_angular_speed_flick)

    def plot_tail(self, ax, line_str, inds_dict):
        # close
        self._plot(ax, line_str, self.time_steps[inds_dict[1:]], self.tail_body_angular_speed_close)
        # flick
        self._plot(ax, line_str, self.time_steps[inds_dict[-1] + np.array([0, self.flick_samples])], self.tail_body_angular_speed_flick)

    def plot_pect(self, ax, line_str, inds_dict):
        # open
        self._plot(ax, line_str, self.time_steps[inds_dict[:2]], self.pect_speed_open)
        # close
        self._plot(ax, line_str, self.time_steps[inds_dict[1:]], self.pect_speed_close)
        # flick
        self._plot(ax, line_str, self.time_steps[inds_dict[-1] + np.array([0, self.flick_samples])], self.pect_speed_flick)


def calc_local_mean_no_outliers(v, filter_size, std_threshold):
    m, s = v.mean(), v.std()
    non_outlier_inds = (v - m) / s < std_threshold
    filtered_v = np.zeros(v.shape)
    for i in range(filter_size, filtered_v.shape[0] - filter_size - 1):
        filtered_v[i] = np.mean(v[i - filter_size:i + filter_size + 1][non_outlier_inds[i - filter_size:i + filter_size + 1]])
    return filtered_v


def calc_angle(v1, v2):
    return np.arccos( np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2) )


def calc_characteristics(_outputs):
    chars = np.zeros((len(_outputs), 4))
    for i, d in enumerate(_outputs):
        # mouth gap
        chars[i, 0] = np.linalg.norm(d[0, :] - d[1, :])

        offset = 0 if np.isnan(d[7, 0]) else 1
        base_vector = d[4, :] - d[8+offset, :]
        head_vector = d[2, :] - d[0, :]
        pect_vector = d[6+offset, :] - d[8+offset, :]
        tail_vector = d[5, :] - d[4, :]
        # head angle
        chars[i, 1] = calc_angle(head_vector, base_vector)
        chars[i, 2] = calc_angle(pect_vector, base_vector)
        chars[i, 3] = calc_angle(tail_vector, base_vector)

    return chars


def calc_max_inds(heatmaps, heatmaps_weight, image_size):
    _output = np.zeros((heatmaps.shape[0], 2))
    for l in range(heatmaps.shape[0]):
        if heatmaps_weight[l] == 1:
            if image_size is not None:
                resized_heatmaps = cv2.resize(heatmaps[l, ...], image_size, interpolation=cv2.INTER_CUBIC)
            else:
                resized_heatmaps = heatmaps[l, ...]
            _output[l, 0] = np.argmax(np.amax(resized_heatmaps, axis=0))
            _output[l, 1] = np.argmax(np.amax(resized_heatmaps, axis=1))
        else:
            _output[l, 0] = np.nan
            _output[l, 1] = np.nan

    return _output


def load_landmark_diff_stats(stats_file='/data/projects/swat/results/eladc/fish_landmarks/dataset_analysis/diff_stats.csv',
                             num_landmarks=10, depth=5):
    with open(stats_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        stat_mat = -np.ones((num_landmarks, 2, depth))
        for i, row in enumerate(csv_reader):
            if i > 0:  # skip header
                stat_mat[i-1, 0, :] = np.array(row[1::4])
                stat_mat[i-1, 1, :] = np.array(row[2::4])
    return stat_mat
