import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def convert_belief_to_output_and_uncertainty(belief):
    b, n, c, h, w = belief.shape
    evidence = F.softplus(belief)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=2, keepdim=True)
    output = alpha / S
    uncertainty = c / S
    return output, uncertainty

def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def preprocess_batch(batch, device, unsqueeze=False):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
            if unsqueeze:
                batch[key] = batch[key].unsqueeze(0)


def set_module_grad(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
