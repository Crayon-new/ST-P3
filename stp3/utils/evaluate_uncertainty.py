import torch
import torch.nn.functional as F
import numpy as np

def get_ce_and_e(labels, output, n_present):
    segmentation_gt = labels['segmentation'][:, n_present - 1].detach()
    semantic_seg_gt = segmentation_gt[0][0].cpu().numpy()

    segmentation = output['segmentation'][:, n_present - 1].detach()
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_seg = semantic_seg > 0

    seg_union = np.logical_or(semantic_seg_gt, semantic_seg)

    pred_union = segmentation[0].permute(1,2,0)[seg_union]
    gt_union = semantic_seg_gt[seg_union]
    ce = F.cross_entropy(pred_union, torch.tensor(gt_union).cuda(), reduction='none').cpu().numpy()
    
    entropy = output['heat_map'][seg_union]
    return np.stack((entropy, ce), axis=1)
