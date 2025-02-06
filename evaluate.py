from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse, convert_belief_to_output_and_uncertainty
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour, plot_prediction
from stp3.config import get_cfg
import matplotlib.cm as cm
from scipy import ndimage
import importlib

def mk_save_dir(debug_mode):
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs')
    if debug_mode:
        save_path = save_path / 'debug_imgs'
    save_path = save_path / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot, debug_mode=False):
    save_path = mk_save_dir(debug_mode)

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=False)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot

    cfg.MODEL.SAMPLE_RESULTS = True

    cfg.DATASET.USE_CORRUPTION = True
    cfg.DATASET.CORRUPTION_TYPE = 'Fog'
    cfg.DATASET.CORRUPTION_LEVEL = 'mid'
    cfg.DATASET.CORRUPTION_DATAROOT = 'data/nuScenes-c'

    cfg.MODEL.TEST_SAMPLE_NUM = 100

    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))


    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = len(image)
        labels = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )

        n_present = model.receptive_field

        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        if cfg.INSTANCE_SEG.ENABLED:
            pred_consistent_instance_seg, matched_centers= predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=True, make_consistent=True
            )
            metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                     labels['instance'][:, n_present - 1:])
            output['prediction_np_result'] = plot_prediction(pred_consistent_instance_seg, matched_centers)

            target_consistent_instance_seg, target_matched_centers= predict_instance_segmentation_and_trajectories(
                labels, compute_matched_centers=True, make_consistent=True
            )
            labels['target_prediction_result'] = plot_prediction(target_consistent_instance_seg, target_matched_centers)

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )
            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                metric_planning_val[i](final_traj[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])

        if index % 100 == 0:
            if cfg.PLANNING.ENABLED:
                output = {**output, 'pred_trajectory': final_traj}
            save(output, labels, batch, n_present, index, save_path)


    results = {}

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    if cfg.INSTANCE_SEG.ENABLED:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')


if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='/home/huangzj/github_repo/ST-P3/tensorboard_logs/06July2024at15_44_02CST_rtxgpu001_Prediction/default/version_0/checkpoints/epoch=18-step=55536.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--debug_mode', default=False, type=bool)
    parser.add_argument('--dataroot', default='data/nuscene', type=str)

    args = parser.parse_args()
    # register module
    importlib.import_module('stp3.models.transformer')
    importlib.import_module('mmdet.models.utils')

    eval(args.checkpoint, args.dataroot, args.debug_mode)
