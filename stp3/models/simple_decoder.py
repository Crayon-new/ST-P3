import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from stp3.layers.convolutions import UpsamplingAdd, DeepLabHead
from stp3.models.feature_distribution import feature_distribution


class Simple_Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, n_present, n_hdmap, predict_gate):
        super().__init__()
        self.perceive_hdmap = predict_gate['perceive_hdmap']
        self.predict_pedestrian = predict_gate['predict_pedestrian']
        self.predict_instance = predict_gate['predict_instance']
        self.predict_future_flow = predict_gate['predict_future_flow']
        self.planning = predict_gate['planning']

        self.n_classes = n_classes
        self.n_present = n_present
        if self.predict_instance is False and self.predict_future_flow is True:
            raise ValueError('flow cannot be True when not predicting instance')

        shared_out_channels = in_channels

        self.feature_distribution = feature_distribution(2, 2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
        )

        if self.predict_pedestrian:
            self.pedestrian_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
            )

        if self.perceive_hdmap:
            self.hdmap_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2 * n_hdmap, kernel_size=1, padding=0),
            )

        if self.predict_instance:
            self.instance_offset_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )
            self.instance_center_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.planning:
            self.costvolume_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        segmentation_output = torch.clamp(self.segmentation_head(x), -6, 6)
        mean_states, sigma_states, segmentation_output = self.feature_distribution(segmentation_output)
        pedestrian_output = self.pedestrian_head(x) if self.predict_pedestrian else None
        hdmap_output = self.hdmap_head(x.view(b, s, *x.shape[1:])[:,self.n_present-1]) if self.perceive_hdmap else None
        instance_center_output = self.instance_center_head(x) if self.predict_instance else None
        instance_offset_output = self.instance_offset_head(x) if self.predict_instance else None
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None
        costvolume = self.costvolume_head(x).squeeze(1) if self.planning else None
        return {
            'segmentation': segmentation_output.view(b, s, *segmentation_output.shape[1:]),
            'mean_states': mean_states,
            'sigma_states': sigma_states,
            'pedestrian': pedestrian_output.view(b, s, *pedestrian_output.shape[1:])
            if pedestrian_output is not None else None,
            'hdmap' : hdmap_output,
            'instance_center': instance_center_output.view(b, s, *instance_center_output.shape[1:])
            if instance_center_output is not None else None,
            'instance_offset': instance_offset_output.view(b, s, *instance_offset_output.shape[1:])
            if instance_offset_output is not None else None,
            'instance_flow': instance_future_output.view(b, s, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
            'costvolume': costvolume.view(b, s, *costvolume.shape[1:])
            if costvolume is not None else None,
        }
