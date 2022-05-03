from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .layers import DualAttentionBlock, RecursiveResidualGroup


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size: int = 21, sigma: float = 5, channels: int = 3):
        """ Gaussian blur layer. """
        super().__init__()

        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                         kernel_size=kernel_size, groups=channels, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        x = self.gaussian_filter(x)
        return x


class BasicPipeline(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 24, n_dab: int = 5, ksize: int = 3):
        """ Basic Pipeline for RGB2RAW. """
        super().__init__()

        self.conv_head = nn.Conv2d(in_channels,  n_feat, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.conv_tail = nn.Conv2d(n_feat, out_channels, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.rrg1 = RecursiveResidualGroup(n_feat,     n_dab, ksize)
        self.rrg2 = RecursiveResidualGroup(n_feat * 4, n_dab, ksize)
        self.rrg3 = RecursiveResidualGroup(n_feat,     n_dab, ksize)
        self.downscale = nn.PixelUnshuffle(2)
        self.upscale   = nn.PixelShuffle(2)

        init.xavier_normal_(self.conv_head.weight)
        init.xavier_normal_(self.conv_tail.weight)

    def forward(self, x: torch.Tensor):
        x = self.conv_head(x)
        x = self.rrg1(x)
        x = self.downscale(x)
        x = self.rrg2(x)
        x = self.upscale(x)
        x = self.rrg3(x)
        x = self.conv_tail(x)
        return x


class ColorFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, n_feat: int = 24, blur_sigma: float = 12, n_dab: int = 2, ksize: int = 3):
        """ Color Correction Branch for CycleISP. """
        super().__init__()

        self.conv_head = nn.Conv2d(in_channels, n_feat,     ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.conv_tail = nn.Conv2d(n_feat * 4,  n_feat * 4, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.rrg1 = RecursiveResidualGroup(n_feat,     n_dab, ksize)
        self.rrg2 = RecursiveResidualGroup(n_feat * 4, n_dab, ksize)
        self.downscale = nn.PixelUnshuffle(2)
        self.blur = GaussianBlur(sigma=blur_sigma, kernel_size=4 * blur_sigma + 1, channels=in_channels)

        init.xavier_normal_(self.conv_head.weight)
        init.xavier_normal_(self.conv_tail.weight)

    def forward(self, x: torch.Tensor):
        x = self.blur(x)
        x = self.conv_head(x)
        x = self.rrg1(x)
        x = self.downscale(x)
        x = self.rrg2(x)
        x = self.conv_tail(x)
        return x


class GuidedPipeline(BasicPipeline):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_feat: int = 24, n_dab: int = 5, ksize: int = 3):
        """ Pipeline with color features. """
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, n_dab=n_dab, ksize=ksize)

    def forward(self, x: torch.Tensor, feat: torch.Tensor):
        x = self.conv_head(x)
        x = self.rrg1(x)
        x = self.downscale(x)
        x = self.rrg2(x)
        x = x + x * feat
        x = self.upscale(x)
        x = self.rrg3(x)
        x = self.conv_tail(x)
        return x
