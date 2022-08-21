import os
import pickle
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


output_folder = '/results/report_final_params'
os.makedirs(output_folder, exist_ok=True)

# folders_list = ['/results/20220511_173611__crop_landmarks_first_max_angle_10_no_squash_random_cropFIS_75',
#                 '/results/20220515_002917__same_as_75_input_576FIS_82',
#                 '/results/20220522_221719__hrnet_576_FIS_99',
#                 '/results/20220526_220731__hrnet_384_report',
#                 '/results/20220607_092851__hrnet_384_output_384',
#                 '/results/20220610_233103__hrnet_384_output_384_sigma_4',
#                 ]
#
# folders_description_list = ['FIS-75, LPN50 384',
#                             'FIS-82, LPN50 576',
#                             'FIS-99, HRNet 576',
#                             'FIS-101, HRNet 384',
#                             'FIS-109, HRNet 384 full res',
#                             'FIS-115, HRNet 384 full res',
#                             ]

folders_list = ['/results/20220610_233103__hrnet_384_output_384_sigma_4',
                '/results/20220616_000229__hrnet_384',
                '/results/20220511_173611__crop_landmarks_first_max_angle_10_no_squash_random_cropFIS_75',
                ]

folders_description_list = ['FIS-115, HRNet 384 full res',
                            'FIS-144, HRNet 384',
                            'FIS-75, LPN50 384',
                            ]

# folders_list = ['/results/20220616_000229__hrnet_384_linear',
#                 '/results/20220616_000229__hrnet_384',
#                 ]
# folders_description_list = ['FIS-144, HRNet 384 - bicubic',
#                             'FIS-144, HRNet 384 - linear',
#                             ]


def plot_multi_bars(_mean_results, _std_results, _title, output_file, mouth_scale=1,
                    xticks=None, xlabels=None, **xtick_kwargs):
    N, M = _mean_results.shape

    if _std_results is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_position([0.1, 0.3, 0.8, 0.6])
        axs = [ax]
    else:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(_title)

    w = 0.8
    x_diff = w/M
    x = np.arange(N+1)-w/2+x_diff/M

    for m in range(M):
        _res = _mean_results[_mean_results[:, m] >= 0, m].mean()
        _mean_results[:2, m] = _mean_results[:2, m] * mouth_scale
        axs[0].bar(x + m * x_diff, np.concatenate([_mean_results[:, m], _res.reshape((1,))]), width=x_diff, label=folders_description_list[m])
        if _std_results is not None:
            _res = _std_results[_std_results[:, m]>0, m].mean()
            _std_results[:2, m] = _std_results[:2, m] * mouth_scale
            axs[1].bar(x + m * x_diff, np.concatenate([_std_results[:, m], _res.reshape((1,))]), width=x_diff, label=folders_description_list[m])
    if xlabels is None:
        xlabels = [str(_) for _ in np.arange(N)] + ['all']
    if mouth_scale == 10:
        xlabels[0] = xlabels[0] + ' [mm]'
        xlabels[1] = xlabels[1] + ' [mm]'
    elif mouth_scale != 1:
        raise Exception('What???')
    if xticks is None:
        xticks = np.arange(N+1)
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xticks(ticks=xticks, labels=xlabels, **xtick_kwargs)
    if _std_results is not None:
        axs[1].grid()
        axs[1].set_xticks(ticks=xticks, labels=xlabels, **xtick_kwargs)

    fig.savefig(output_file)
    plt.close()


def plot_hists(_hists, _bins, _title, _file):
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle(_title)

    for _hist in _hists:
        axs[2, 3].plot((_bins[:-1] + _bins[1:]) / 2, _hist[0][0])
    axs[2, 3].set_title('all landmarks')
    axs[2, 3].grid()

    for landmark_idx in range(10):
        i, j = int(landmark_idx / 4), landmark_idx % 4
        for m, _hist in enumerate(_hists):
            _values, _mean, _std = _hist[landmark_idx + 1]
            if _values is not None:
                axs[i, j].plot((_bins[:-1] + _bins[1:]) / 2, _values, label=folders_description_list[m])
        if _values is not None:
            axs[i, j].set_title(f'#{landmark_idx}: mean={_mean: 2.2f},  std={_std: 2.2f}')
            axs[i, j].grid()
        axs[0, 0].legend()

    fig.savefig(_file)
    plt.close()


if __name__ == '__main__':
    reports, data, final_params = [], [], []
    for folder in folders_list:
        with open(join(folder, 'report.pickle'), 'rb') as f:
            reports.append(pickle.load(f))
            # _data[vid_name] = {'all_2d': [mean_2d, std_2d],
            #                    '2d': stat_2d_per_landmark,
            #                    'all_3d': [mean_3d, std_3d],
            #                    '3d': stat_3d_per_landmark,
            #                    }
        with open(join(folder, 'data.pickle'), 'rb') as f:
            data.append(pickle.load(f))
        # {'model_output_3d': model_output_3d,
        #              'model_output_3d_dlt': model_output_3d_dlt,
        #              'labels_3d': labels_3d,
        #              'model_output_2d': model_output_2d,
        #              'labels_2d': labels_2d,
        #              'dlt_geometric_diff': dlt_geometric_diff,
        #              'res': res
        #  }
        with open(join(folder, 'report_final_params.pickle'), 'rb') as f:
            final_params.append(pickle.load(f))
        # final_params_dict[vid_name] = [label_final_params, final_params]

    total_mean_results = np.zeros((10, len(folders_list), len(reports[0])))
    total_std_results = np.zeros((10, len(folders_list), len(reports[0])))
    final_params_names = ['gape_opening_speed', 'gape_closing_speed', 'gape_flick_speed',
                          'head_body_angular_speed_close', 'head_body_angular_speed_flick',
                          'tail_body_angular_speed_close', 'tail_body_angular_speed_flick',
                          'pect_speed_open', 'pect_speed_close', 'pect_speed_flick',
                          'body_speed_open', 'body_speed_close', 'body_speed_flick']
    final_params_results = np.zeros((len(final_params_names), len(folders_list), len(reports[0])))
    for vid_index, vid_name in enumerate(reports[0]):
        for i, final_params_report in enumerate(final_params):
            for j, fp_name in enumerate(final_params_names):
                label_param = final_params_report[vid_name][0].__getattribute__(fp_name)[0]
                measured_param = final_params_report[vid_name][1].__getattribute__(fp_name)[0]
                final_params_results[j, i, vid_index] = np.abs(label_param-measured_param)
        plot_multi_bars(final_params_results[..., vid_index], None,
                        f'{vid_name} final params', join(output_folder, f'final_params_{vid_name}.png'),
                        xticks=np.arange(len(final_params_names)), xlabels=final_params_names,
                        rotation='vertical')

        num_landmarks = reports[0][vid_name]['3d'].shape[0]
        mean_results = np.zeros((num_landmarks, len(folders_list)))
        std_results = np.zeros((num_landmarks, len(folders_list)))
        mean_results_2d = np.zeros((num_landmarks, len(folders_list)))
        std_results_2d = np.zeros((num_landmarks, len(folders_list)))
        for i, (report, folder_description) in enumerate(zip(reports, folders_description_list)):
            mean_results[:, i] = report[vid_name]['3d'][:, 0]
            std_results[:, i] = report[vid_name]['3d'][:, 1]
            mean_results_2d[:, i] = report[vid_name]['2d'][:, 0]
            std_results_2d[:, i] = report[vid_name]['2d'][:, 1]
            total_mean_results[:, i, vid_index] = mean_results[:, i]
            total_std_results[:, i, vid_index] = std_results[:, i]
        mean_results[mean_results<0] = 0
        std_results[std_results<0] = 0
        mean_results_2d[mean_results_2d<0] = 0
        std_results_2d[std_results_2d<0] = 0

        plot_multi_bars(mean_results, std_results, f'{vid_name} 3d error comparison [cm]', join(output_folder, f'err_3d_{vid_name}.png'), mouth_scale=10)
        plot_multi_bars(mean_results_2d, std_results_2d, f'{vid_name} 2d error comparison [pixel]', join(output_folder, f'err_2d_{vid_name}.png'))

        hist_2d_list, bins_2d, hist_3d_list, bins_3d = [], None, [], None
        for i, (_data, folder_description) in enumerate(zip(data, folders_description_list)):
            hist_2d, hist_3d = [], []
            model_output_3d = data[i]['model_output_3d']
            model_output_3d_dlt = data[i]['model_output_3d_dlt']
            labels_3d = data[i]['labels_3d']
            model_output_2d = data[i]['model_output_2d']
            labels_2d = data[i]['labels_2d']
            dlt_geometric_diff = data[i]['dlt_geometric_diff']
            res = data[i]['res']

            # 2D
            _label = np.stack(labels_2d[vid_name])
            _out = np.stack(model_output_2d[vid_name])
            _diff = np.linalg.norm(_label - _out, axis=2)
            hist_range = (0, 50)
            values_2d = _diff.flatten()[~np.isnan(_diff.flatten())]
            mean_2d, std_2d = values_2d.mean(), values_2d.std()
            hist, bins_2d = np.histogram(values_2d, bins=100, range=hist_range)
            hist_2d.append([hist, mean_2d, std_2d])
            for landmark_idx in range(_diff.shape[1]):
                _values = _diff[~np.isnan(_diff[:, landmark_idx]), landmark_idx]
                if len(_values):
                    _values[_values > hist_range[1]] = hist_range[1]
                    m, s = _values.mean(), _values.std()
                    hist, _ = np.histogram(_values, bins=100, range=hist_range)
                else:
                    hist, m, s = None, None, None
                hist_2d.append([hist, m, s])

            # 3D
            _label = np.stack(labels_3d[vid_name])
            _out = np.stack(model_output_3d[vid_name])
            _diff = np.linalg.norm(_label - _out, axis=2)
            hist_range = (0, 5)
            values_3d = _diff.flatten()[~np.isnan(_diff.flatten())]
            mean_3d, std_3d = values_3d.mean(), values_3d.std()
            hist, bins_3d = np.histogram(values_3d, bins=100, range=hist_range)
            hist_3d.append([hist, mean_3d, std_3d])
            for landmark_idx in range(_diff.shape[1]):
                _values = _diff[~np.isnan(_diff[:, landmark_idx]), landmark_idx]
                if len(_values):
                    _values[_values > hist_range[1]] = hist_range[1]
                    m, s = _values.mean(), _values.std()
                    hist, _ = np.histogram(_values, bins=100, range=hist_range)
                else:
                    hist, m, s = None, None, None
                hist_3d.append([hist, m, s])

            hist_2d_list.append(hist_2d)
            hist_3d_list.append(hist_3d)

        plot_hists(hist_2d_list, bins_2d, f'{vid_name} 2d error histograms [pixel]', join(output_folder, f'hist_diff2d_{vid_name}.png'))
        plot_hists(hist_3d_list, bins_3d, f'{vid_name} 3d error histograms [cm]', join(output_folder, f'hist_diff3d_{vid_name}.png'))

    for landmark_idx in range(total_mean_results.shape[0]):
        for report_index in range(total_mean_results.shape[1]):
            v = total_mean_results[landmark_idx, report_index, :]
            total_mean_results[landmark_idx, report_index, 0] = v[v>=0].mean()
            v = total_std_results[landmark_idx, report_index, :]
            total_std_results[landmark_idx, report_index, 0] = v[v >= 0].mean()
    plot_multi_bars(total_mean_results[:, :, 0], total_std_results[:, :, 0], 'total: 3d error comparison [cm]', join(output_folder, 'err_3d.png'), mouth_scale=10)
