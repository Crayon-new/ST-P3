from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
import matplotlib
from matplotlib import pyplot as plt
from stp3.utils.network import NormalizeInverse
from stp3.utils.visualisation import make_contour

import numpy as np
import torch.nn
from sklearn.metrics import *

import cv2, os

colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])

def patch_metrics(uncertainty_scores, uncertainty_labels, sample_size=1_000_000):
    thresholds = np.linspace(0, 1, 10)
    pavpus = []
    agcs = []
    ugis = []

    for threshold in thresholds:
        pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=threshold)
        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2, window_size=4):
    ac, ic, au, iu = 0., 0., 0., 0.

    anchor = (0, 0)
    last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

    while anchor != last_anchor:
        label_window = uncertainty_labels[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]
        uncertainty_window = uncertainty_scores[:, anchor[0]:anchor[0] + window_size, anchor[1]:anchor[1] + window_size]

        accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
        avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

        accurate = accuracy < accuracy_threshold
        uncertain = avg_uncertainty >= uncertainty_threshold

        au += torch.sum(accurate & uncertain)
        ac += torch.sum(accurate & ~uncertain)
        iu += torch.sum(~accurate & uncertain)
        ic += torch.sum(~accurate & ~uncertain)

        if anchor[1] < uncertainty_labels.shape[1] - window_size:
            anchor = (anchor[0], anchor[1] + window_size)
        else:
            anchor = (anchor[0] + window_size, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu.item(), a_given_c.item(), u_given_i.item()


def roc_pr(uncertainty_scores, uncertainty_labels, sample_size=1_000_000, seq = -1):
    assert seq < 3 and seq >-2
    if seq != -1:
        y_true = uncertainty_labels[:, seq, :, :].cpu().flatten()
        y_score = uncertainty_scores[:, seq, :, :].cpu().flatten()
    else:
        y_true = uncertainty_labels.cpu().flatten()
        y_score = uncertainty_scores.cpu().flatten()

    indices = np.random.choice(y_true.shape[0], sample_size, replace=False)

    y_true = y_true[indices]
    y_score = y_score[indices]

    pr, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)

    no_skill = torch.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill


def parse(imgs, gt):
    seg = imgs[:, :, 3:, :, :].view(-1, 3, 128, 352)
    back = ~(seg[:, 0, :, :] + seg[:, 1, :, :] + seg[:, 2, :, :]).bool()[:, None, :, :]
    seg = torch.cat((seg, back), dim=1)
    i = imgs[:, :, :3, :, :]

    if gt:
        back = ~(imgs[:, :, 3, :, :] + imgs[:, :, 4, :, :] + imgs[:, :, 5, :, :]).bool()[:, :, None, :, :]

        return torch.cat((imgs, back), dim=2), seg

    return i, seg


def get_iou(preds, labels):
    classes = preds.shape[1]
    intersect = [0]*classes
    union = [0]*classes

    with torch.no_grad():
        for i in range(classes):
            pred = (preds[:, i, :, :] >= .5)
            tgt = labels[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return intersect, union

def map(img, m=False):
    if not m:
        dense = img.detach().cpu().numpy().argmax(-1)
    else:
        dense = img.detach().cpu().numpy()

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color
    return rgb


def save_pred(pred, labels, out_path):
    pred = map(pred[0].permute(1, 2, 0))
    labels = map(labels[0].permute(1, 2, 0))

    cv2.imwrite(os.path.join(out_path, "pred.jpg"), pred)
    cv2.imwrite(os.path.join(out_path, "label.jpg"), labels)

    return pred, labels


def save_stp3(output, labels, batch, n_present, frame, save_path):
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()
    gt_trajs = labels['gt_trajectory']
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(4*val_w,2*val_h))
    width_ratios = (val_w,val_w,val_w,val_w)
    gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0,n_present-1,3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 2])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[:, 3])
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    hdmap = labels['hdmap'].detach()
    # lane
    area = hdmap[0, 0:2][1].cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # drivable
    area = hdmap[0, 0:2][0].cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # # drivable
    # area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # # lane
    # area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(make_contour(showing))
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    # gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    # gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    # plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0, c='b', alpha=0.2)

    if 'pred_trajectory' in output:
        gt_trajs = output['pred_trajectory'].detach().cpu()
    # add self point
    gt_trajs = torch.cat([torch.zeros((1, 1, 3)), gt_trajs], dim=1)
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0, c='r', alpha=1) 

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()

def save_robio(output, labels, batch, n_present, frame, save_path):
    gt_trajs = labels['gt_trajectory'].cpu()
    if 'pred_trajectory' in output:
        gt_trajs = output['pred_trajectory'].detach().cpu()
    # add self point
    gt_trajs = torch.cat([torch.zeros((1, 1, 3)), gt_trajs], dim=1)
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(3 * val_w, 6 * val_h))
    width_ratios = (val_w, val_w, val_w)
    gs = matplotlib.gridspec.GridSpec(6, 3, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0, n_present - 1, 0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0, n_present - 1, 1].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0, n_present - 1, 2].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 2])
    plt.annotate('BACK_RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    # plt.subplot(gs[2:4, 0])
    # showing = torch.zeros((200, 200, 3)).numpy()
    # showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # # drivable
    # if output['hdmap'] is not None:
    #     hdmap = output['hdmap'].detach()
    #     area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    #     hdmap_index = area > 0
    #     showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    #     # lane
    #     area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    #     hdmap_index = area > 0
    #     showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])
    # else:
    #     hdmap = labels['hdmap'].detach()
    #     # lane
    #     area = hdmap[0, 0:2][1].cpu().numpy()
    #     hdmap_index = area > 0
    #     showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    #     # drivable
    #     area = hdmap[0, 0:2][0].cpu().numpy()
    #     hdmap_index = area > 0
    #     showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    if output['segmentation'] is not None:
        output['segmentation'], output['seg_uncertainty'] = convert_belief_to_output_and_uncertainty(output['segmentation'])
        segmentation = output['segmentation'][:, n_present - 1].detach()
        seg_uncertainty = output['seg_uncertainty'][:, n_present - 1].detach()
        # semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
        # semantic_index = semantic_seg > 0
        # showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    # if output['pedestrian'] is not None:
    #     pedestrian = output['pedestrian'][:, n_present - 1].detach()
    #     pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    #     pedestrian_index = pedestrian_seg > 0
    #     showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    # plt.imshow(make_contour(showing))
    # plt.axis('off')

    bx = np.array([-50.0 + 0.5 / 2.0, -50.0 + 0.5 / 2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    # plt.xlim((200, 0))
    # plt.ylim((0, 200))
    # gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    # gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    # plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    if 'sigma_states' in output:

        # sigma
        plt.subplot(gs[4:6, 0])
        plt.annotate('U(t+1)', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        # seg_uncertainty = output['seg_uncertainty'][0].detach().cpu().numpy()
        seg_uncertainty = output['UQ'].detach().cpu().numpy()
        n_present -= 3 # past uncertainty
        seg_uncertainty = np.mean(seg_uncertainty[n_present-1], axis=0)
        cmap = cm.ScalarMappable(cmap='viridis')
        colormap_array = cmap.to_rgba(seg_uncertainty)[:,:,:3]
        plt.imshow(make_contour(colormap_array))
        plt.axis('off')
        # plt.colorbar()


        plt.fill(pts[:, 0], pts[:, 1], '#76b900')
        plt.xlim((200, 0))
        plt.ylim((0, 200))
        # sigma
        plt.subplot(gs[4:6, 1])
        plt.annotate('U(t+2)', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        # seg_uncertainty = output['seg_uncertainty'][0].detach().cpu().numpy() # future uncertainty
        seg_uncertainty = output['UQ'].detach().cpu().numpy()
        seg_uncertainty = np.mean(seg_uncertainty[n_present], axis=0)

        cmap = cm.ScalarMappable(cmap='viridis')
        colormap_array = cmap.to_rgba(seg_uncertainty)[:,:,:3]
        plt.imshow(make_contour(colormap_array))
        plt.axis('off')
        # plt.colorbar()


        plt.fill(pts[:, 0], pts[:, 1], '#76b900')
        plt.xlim((200, 0))
        plt.ylim((0, 200))
        # sigma
        plt.subplot(gs[4:6, 2])
        plt.annotate('U(t+3)', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        # seg_uncertainty = output['seg_uncertainty'][0].detach().cpu().numpy()
        seg_uncertainty = output['UQ'].detach().cpu().numpy()
        seg_uncertainty = np.mean(seg_uncertainty[n_present+1], axis=0)
        cmap = cm.ScalarMappable(cmap='viridis')
        colormap_array = cmap.to_rgba(seg_uncertainty)[:,:,:3]
        plt.imshow(make_contour(colormap_array))
        plt.axis('off')
        # plt.colorbar()


        plt.fill(pts[:, 0], pts[:, 1], '#76b900')
        plt.xlim((200, 0))
        plt.ylim((0, 200))

    if 'target_prediction_result' in labels:
        plt.subplot(gs[2:4, 1])
        plt.annotate('GT', (0.01, 0.87), c='black', xycoords='axes fraction', fontsize=14)
        plt.imshow(make_contour(labels['target_prediction_result']))
        plt.axis('off')

        plt.fill(pts[:, 0], pts[:, 1], '#76b900')

        plt.xlim((200, 0))
        plt.ylim((0, 200))

    # # groud truth representations
    # hdmap = labels['hdmap'].detach()

    # plt.subplot(gs[2:4, 1])
    # showing = torch.zeros((200, 200, 3)).numpy()
    # showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # # lane
    # area = hdmap[0, 0:2][1].cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # # drivable
    # area = hdmap[0, 0:2][0].cpu().numpy()
    # hdmap_index = area > 0
    # showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # # semantic
    # segmentation = labels['segmentation'][:, n_present - 1].detach()
    # semantic_seg = segmentation[0][0].cpu().numpy()
    # semantic_index = semantic_seg > 0
    # showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    # if 'pedestrain' in labels and labels['pedestrian'] is not None:
    #     pedestrian = labels['pedestrian'][:, n_present - 1].detach()
    #     pedestrian_seg = pedestrian[0][0].cpu().numpy()
    #     pedestrian_index = pedestrian_seg > 0
    #     showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    # plt.imshow(make_contour(showing))
    # plt.axis('off')

    # plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    # plt.xlim((200, 0))
    # plt.ylim((0, 200))

    # plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    if 'prediction_np_result' in output:
        plt.subplot(gs[2:4, 0])
        plt.annotate('Pred', (0.01, 0.87), c='black', xycoords='axes fraction', fontsize=14)
        plt.imshow(make_contour(output['prediction_np_result'][::-1, ::-1]))
        plt.axis('off')
        plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()
