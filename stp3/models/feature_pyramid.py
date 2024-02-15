import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from stp3.layers.convolutions import UpsamplingAdd, DeepLabHead

class Fpn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)


    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  # (b*s, 256, 25, 25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        x = x.view(b, s, *x.shape[1:])

        return x