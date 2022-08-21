import pickle
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms

from datasets.dataset_2d import Dataset2D
import datasets.transforms as np_transforms

from test import parse_args
from lib.config import cfg
from lib.config import update_config
import lib.models as models
from lib.core.loss import JointsMSELoss
from lib.core.function import AverageMeter, accuracy
from lib.utils.utils import get_optimizer
from user_constants_mine import OUTPUT_FOLDER, DEFAULT_DATASET_PATH


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    if args.neptune:
        # Create run in project
        import neptune.new as neptune
        from user_constants import _PROJECT_NAME, _NEPTUNE_API_TOKEN

        run = neptune.init(project=_PROJECT_NAME,
                           api_token=_NEPTUNE_API_TOKEN,
                           )

    if cfg.MODEL.NAME.lower() == 'lpn':
        get_model = models.lpn.get_pose_net
    elif cfg.MODEL.NAME.lower() == 'pose_hrnet':
        get_model = models.pose_hrnet.get_pose_net
    else:
        raise NotImplemented
    model = get_model(cfg, is_train=True)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
                              gamma=cfg.LOSS.FL_GAMMA, focal_loss_enable=cfg.LOSS.USE_FOCAL_LOSS,
                              focal_temp=cfg.LOSS.FOCAL_TEMP)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = get_optimizer(cfg, model)
    scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    thr = [3, 3,  # mouth
           5, 10,  # forehead, back
           5, 5,  # tail
           4, 4,  # pectoral - base
           4, 4,  # pectoral - tip
           ]

    ds_path = cfg.DATASET.ROOT if os.path.exists(cfg.DATASET.ROOT) else DEFAULT_DATASET_PATH
    print('Dataset path is', ds_path)
    VAL_SPLIT = [0, 4, 12, 16, 21, 26, 33, 39]

    train_np_transforms = [np_transforms.FlipLeftRight(landmark_flip=cfg.DATASET.FLIP_INDS, num_input_images=cfg.MODEL.NUM_INPUT_IMAGES),
                           # np_transforms.CropLandmarks(ratio=3.0),
                           np_transforms.RandomRotateScale(max_angle=cfg.DATASET.MAX_ANGLE,
                                                           scale_limits=cfg.DATASET.SCALE_LIMITS)]
    if cfg.DATASET.SQUASH_SHIFT > 0:
        train_np_transforms.append(np_transforms.RandomSquash(cfg.DATASET.SQUASH_SHIFT))
    train_np_transforms.extend([#np_transforms.CropLandmarks(ratio=cfg.DATASET.CROP_RATIO, random_crop=False),
                                np_transforms.Resize(image_size=cfg.MODEL.IMAGE_SIZE)])
    if cfg.DATASET.CUTOUT_PROB > 0:
        train_np_transforms.append(np_transforms.CutOut(cutout_prob=cfg.DATASET.CUTOUT_PROB,
                                                        min_cutout=cfg.DATASET.CUTOUT_MIN,
                                                        max_cutout=cfg.DATASET.CUTOUT_MAX))
    if True:#cfg.DATASET.CUTOUT_PROB > 0:
        train_np_transforms.append(np_transforms.Awgn(snr_db=15))

    train_dataset = Dataset2D(train=True, val_split=VAL_SPLIT, sigma=cfg.MODEL.SIGMA, imgs_folder=ds_path,
                              image_size=cfg.MODEL.IMAGE_SIZE[0], heatmap_size=cfg.MODEL.HEATMAP_SIZE,
                              joints_weight=cfg.MODEL.JOINTS_WEIGHT, num_input_images=cfg.MODEL.NUM_INPUT_IMAGES,
                              use_prev_hm_input=cfg.MODEL.USE_PREV_HM_INPUT,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(127.5, 127.5)]),
                              np_transforms=train_np_transforms,
                              fixed_crop_per_vid=True, fixed_crop_ratio=cfg.DATASET.CROP_RATIO)
    valid_dataset = Dataset2D(train=False, val_split=VAL_SPLIT, sigma=cfg.MODEL.SIGMA, imgs_folder=ds_path,
                              joints_weight=cfg.MODEL.JOINTS_WEIGHT, num_input_images=cfg.MODEL.NUM_INPUT_IMAGES,
                              use_prev_hm_input=cfg.MODEL.USE_PREV_HM_INPUT,
                              image_size=cfg.MODEL.IMAGE_SIZE[0], heatmap_size=cfg.MODEL.HEATMAP_SIZE,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(127.5, 127.5)]),
                              np_transforms=[#np_transforms.CropLandmarks(ratio=cfg.DATASET.VAL_CROP_RATIO),
                                             np_transforms.Resize(image_size=cfg.MODEL.IMAGE_SIZE),
                                             ],
                              fixed_crop_per_vid=True, fixed_crop_ratio=cfg.DATASET.VAL_CROP_RATIO)

    num_gpus = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*num_gpus,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*num_gpus,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    comment = '' if args.comment is None else ' '.join(args.comment)
    time_stamp = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}__{comment.replace(" ", "_")}'
    # results_folder = os.path.join(os.environ.get('SWAT'), 'results', os.environ.get('USER'),
    #                               'fish_landmarks', time_stamp)
    results_folder = OUTPUT_FOLDER
    print('results folder:', results_folder)
    os.makedirs(results_folder, exist_ok=True)
    shutil.copy2(args.cfg, results_folder)

    if args.neptune:
        optimizer_config = optimizer.param_groups[0].copy()
        _ = optimizer_config.pop('params')
        run["parameters"] = {"optimizer_params": optimizer_config,
                             "optimizer": type(optimizer).__name__,
                             "results_folder": results_folder,
                             'cfg': cfg,
                             'VAL_SPLIT': VAL_SPLIT,
                             'accuracy_threshold': thr,
                             'comment': comment}

    # Training
    best_acc = 0
    train_time = AverageMeter()
    train_acc_time = AverageMeter()
    val_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_per_landmark = [AverageMeter() for _ in range(10)]
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    val_acc_per_landmark = [AverageMeter() for _ in range(10)]

    num_epochs = cfg.TRAIN.END_EPOCH
    for epoch in range(num_epochs):
        train_time.reset()
        train_acc_time.reset()
        val_time.reset()
        batch_time.reset()
        data_time.reset()
        losses.reset()
        acc.reset()
        val_losses.reset()
        val_acc.reset()

        train_stats_collector = []
        epoch_start = time.time()
        end = time.time()
        model.train()
        if True:  # Set False to skip training
            for i, (data, target, target_weight, pp_data) in enumerate(train_loader):
                t1 = time.time()
                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                if False:  # symmetric batch
                    data_flipped = torch.flip(data, [3])
                    target_flipped = torch.flip(target[:, train_dataset._LANDMARK_FLIP, :, :], [3])
                    target_weight_flipped = target_weight[:, train_dataset._LANDMARK_FLIP, :]
                    data = torch.concat([data, data_flipped], 0)
                    target = torch.concat([target, target_flipped], 0)
                    target_weight = torch.concat([target_weight, target_weight_flipped], 0)

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda(non_blocking=True)
                    target_weight = target_weight.cuda(non_blocking=True)

                if cfg.MODEL.USE_PREV_HM_INPUT:
                    prev_target = target[:, 10:, :, :]
                    target = target[:, :10, :, :]
                    target_weight = target_weight[:, :10, :]
                    outputs = model(data, prev_target)
                else:
                    if cfg.MODEL.NUM_INPUT_IMAGES > 1 and not cfg.MODEL.FINE_TUNE:
                        outputs, inter_hm = model(data, return_inter_hm=True)
                    else:
                        outputs = model(data)

                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    if cfg.MODEL.NUM_INPUT_IMAGES > 1:
                        _inds = ((cfg.MODEL.NUM_INPUT_IMAGES+1)/2*cfg.MODEL.NUM_JOINTS + np.arange(cfg.MODEL.NUM_JOINTS)).astype(np.int32)
                        _target = target[:, _inds, :, :]
                        loss = criterion(output, _target, target_weight)
                        if not cfg.MODEL.FINE_TUNE:
                            loss = (loss + criterion(inter_hm, target, target_weight))/2
                    else:
                        loss = criterion(output, target, target_weight)
                        _target = target

                # compute gradient and do update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_time.update(time.time() - t1)

                t1 = time.time()
                # measure accuracy and record loss
                losses.update(loss.item(), data.size(0))

                _acc_per_landmark, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), _target.detach().cpu().numpy(), thr=thr)
                acc.update(avg_acc, cnt)
                for i_acc, res in zip(acc_per_landmark, _acc_per_landmark):
                    if len(res) > 0:
                        i_acc.update(res.mean(), len(res))

                if False:  # TODO: check if topk can make it run faster
                    # calc absolute landmarks distance
                    _hm = output.detach().cpu().numpy()
                    _label = _target.detach().cpu().numpy()
                    xp = np.argmax(np.amax(_hm, axis=2), axis=2)
                    yp = np.argmax(np.amax(_hm, axis=3), axis=2)
                    xt = np.argmax(np.amax(_label, axis=2), axis=2)
                    yt = np.argmax(np.amax(_label, axis=3), axis=2)
                    dists = np.linalg.norm([xt - xp, yt - yp], axis=0)
                    dists[target_weight.cpu().detach().numpy()[:, :cfg.MODEL.NUM_JOINTS, 0] == 0] = -1
                    train_stats_collector.append(dists)
                train_acc_time.update(time.time() - t1)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            scheduler.step()
        model.eval()
        with torch.no_grad():
            stats_collector = []
            for i, (data, target, target_weight, pp_data) in enumerate(valid_loader):
                # compute output
                t1 = time.time()

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda(non_blocking=True)
                    target_weight = target_weight.cuda(non_blocking=True)

                if cfg.MODEL.USE_PREV_HM_INPUT:
                    prev_target = target[:, 10:, :, :]
                    target = target[:, :10, :, :]
                    target_weight = target_weight[:, :10, :]
                    outputs = model(data, prev_target)
                else:
                    if cfg.MODEL.NUM_INPUT_IMAGES > 1 and not cfg.MODEL.FINE_TUNE:
                        outputs, inter_hm = model(data, return_inter_hm=True)
                    else:
                        outputs = model(data)

                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    if cfg.MODEL.NUM_INPUT_IMAGES > 1:
                        _inds = ((cfg.MODEL.NUM_INPUT_IMAGES + 1) / 2 * cfg.MODEL.NUM_JOINTS + np.arange(cfg.MODEL.NUM_JOINTS)).astype(np.int32)
                        _target = target[:, _inds, :, :]
                        loss = criterion(output, _target, target_weight)
                        if not cfg.MODEL.FINE_TUNE:
                            loss = (loss + criterion(inter_hm, target, target_weight))/2
                    else:
                        loss = criterion(output, target, target_weight)
                        _target = target

                # measure accuracy and record loss
                val_losses.update(loss.item(), data.size(0))

                _acc_per_landmark, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), _target.detach().cpu().numpy(), thr=thr)
                val_acc.update(avg_acc, cnt)
                for i_acc, res in zip(val_acc_per_landmark, _acc_per_landmark):
                    if len(res) > 0:
                        i_acc.update(res.mean(), len(res))

                if False:
                    # calc absolute landmarks distance
                    _hm = output.detach().cpu().numpy()
                    _label = _target.detach().cpu().numpy()
                    xp = np.argmax(np.amax(_hm, axis=2), axis=2)
                    yp = np.argmax(np.amax(_hm, axis=3), axis=2)
                    xt = np.argmax(np.amax(_label, axis=2), axis=2)
                    yt = np.argmax(np.amax(_label, axis=3), axis=2)
                    dists = np.linalg.norm([xt-xp, yt-yp],axis=0)
                    dists[target_weight.cpu().detach().numpy()[:, :cfg.MODEL.NUM_JOINTS, 0] == 0] = -1
                    stats_collector.append(dists)

                val_time.update(time.time() - t1)

        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            # save model
            torch.save(model.state_dict(), os.path.join(results_folder, 'state_dict.pth'))

        epoch_time = time.time() - epoch_start
        if args.neptune:
            run["profiler/train_time"].log(train_time.sum)
            run["profiler/train_acc_time"].log(train_acc_time.sum)
            run["profiler/val_time"].log(val_time.sum)
            run["profiler/data_time"].log(epoch_time - val_time.sum - train_time.sum - train_acc_time.sum)
            run["profiler/epoch_time"].log(epoch_time)
            run["train/loss"].log(losses.avg)
            run["train/acc"].log(acc.avg)
            for i_acc, landmark_acc in enumerate(acc_per_landmark):
                run[f"train/landmark{i_acc}_acc"].log(landmark_acc.avg)
            run["val/loss"].log(val_losses.avg)
            run["val/acc"].log(val_acc.avg)
            for i_acc, landmark_acc in enumerate(val_acc_per_landmark):
                run[f"val/landmark{i_acc}_acc"].log(landmark_acc.avg)
            if len(train_stats_collector) > 0:
                dists = np.concatenate(train_stats_collector, 0).transpose()
                for i, d in enumerate(dists):
                    run[f"train/landmark{i}_avg"].log(d[d > 0].mean())
                    run[f"train/landmark{i}_std"].log(d[d > 0].std())
            if len(stats_collector) > 0:
                dists = np.concatenate(stats_collector, 0).transpose()
                for i, d in enumerate(dists):
                    run[f"val/landmark{i}_avg"].log(d[d > 0].mean())
                    run[f"val/landmark{i}_std"].log(d[d > 0].std())
        print(f'Epoch: [{epoch}]\t'
              f'Avg Batch Time {batch_time.avg:.3f}s\t'
              f'Epoch Time {batch_time.sum:.3f}s\t'
              f'Data {data_time.avg:.3f}s\t'
              f'Loss {losses.avg:.5f} (val loss {val_losses.avg:.5f})\t'
              f'Accuracy {acc.avg:.3f} (val_acc {val_acc.avg:.5f})')

    if args.neptune:
        run.stop()

    print('Done')
