import torch
import torch.nn as nn

class feature_distribution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mean_conv = nn.Sequential(
                                nn.Conv2d(self.in_channels, self.out_channels, 1),
                                nn.BatchNorm2d(self.out_channels)
                            )
        self.sigma_conv = nn.Sequential(
                                        nn.Conv2d(self.in_channels, self.out_channels, 1),
                                        nn.BatchNorm2d(self.out_channels)
                                    )

    def forward(self, states):
        n, c, h, w = states.shape
        mean_states = self.mean_conv(states)
        sigma_states = torch.clamp(self.sigma_conv(states), -7, 7)
        sample_states = mean_states + sigma_states.mul(0.5).exp_() * torch.randn_like(mean_states)
        return mean_states, sigma_states, sample_states