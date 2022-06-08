from typing import Callable, Literal
from collections import OrderedDict
import itertools
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Reshape(nn.Module):
    """ Reshape a tensor. """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 *, init_weights: Callable = init.kaiming_normal_, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2, **kwargs)
        init_weights(self.weight)


class Linear(nn.Linear):
    def __init__(self, in_channels: int, out_channels: int,
                 *, init_weights: Callable = init.kaiming_normal_, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        init_weights(self.weight)


class HyperConv2dKernelNet(nn.Module):
    def __init__(self, h_embeddings: int, in_channels: int, out_channels: int, kernel_size: int,
                 *, mid_channels: int = 256, n_layers: int = 8):
        """ Embed hyperparameters to a convolution kernel. """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # conv weight: (out_channels, in_channels, kernel_size, kernel_size)
        # conv bias:   (out_channels)

        layers = []
        layers.append(Linear(h_embeddings, mid_channels))
        layers.append(nn.LeakyReLU())
        for _ in range(1, n_layers - 1):
            layers.append(Linear(mid_channels, mid_channels))
            layers.append(nn.LeakyReLU())
        layers.append(Linear(mid_channels, 2 * (in_channels + out_channels + kernel_size ** 2),
                             init_weights=init.xavier_normal_))
        self.layers = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights: torch.Tensor = self.layers(h).flatten()

        cut = self.in_channels + self.out_channels + self.kernel_size ** 2
        alpha = self.compose_kernel(weights[:cut])
        beta  = self.compose_kernel(weights[cut:])

        return alpha, beta

    def compose_kernel(self, weight: torch.Tensor) -> torch.Tensor:
        """ Compose a convolution kernel from a weight tensor. """
        cut1 = self.out_channels
        cut2 = cut1 + self.in_channels

        gamma, phi, psi = weight[:cut1], weight[cut1:cut2], weight[cut2:]
        return (gamma.view(self.out_channels, 1, 1, 1) *
                phi.view(1,    self.in_channels, 1, 1) *
                psi.view(1, 1, self.kernel_size, self.kernel_size))


class HyperConv2d(Conv2d):
    def __init__(self, h_embeddings: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        """ Embed hyperparameters to a convolution. """
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.affine_weights = HyperConv2dKernelNet(h_embeddings, in_channels, out_channels, kernel_size)

    def hypertrain(self, enable: bool):
        """ Enable/disable hyperparameter training. """
        for param in self.affine_weights.parameters():
            param.requires_grad = enable
        self.weight.requires_grad = not enable
        self.bias.requires_grad   = not enable

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            return self._conv_forward(x, self.weight, self.bias)
        else:
            y = []

            for b in range(x.size(0)):
                alpha, beta = self.affine_weights(h[b])
                weight = (1 + alpha) * self.weight + beta

                y.append(self._conv_forward(x[b:b + 1], weight, self.bias))

            return torch.cat(y, dim=0)


class ChannelAttn(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 2):
        """ Channel attention layer. """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1   = Conv2d(n_channels, n_channels // reduction, 1)
        self.relu    = nn.LeakyReLU()
        self.conv2   = Conv2d(n_channels // reduction, n_channels, 1, init_weights=init.xavier_normal_)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        s = self.avgpool(x)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        return x * s


class SpatialAttn(nn.Module):
    def __init__(self, kernel_size: int = 3):
        """ Spatial attention layer. """
        super().__init__()
        self.conv    = Conv2d(2, 1, kernel_size, padding_mode="reflect", init_weights=init.xavier_normal_)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        avgpool = torch.mean(x, dim=1, keepdim=True)
        maxpool = torch.max(x,  dim=1, keepdim=True)[0]
        s = torch.cat([avgpool, maxpool], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


class HyperDualAttn(nn.Module):
    def __init__(self, h_embeddings: int, n_channels: int, kernel_size: int = 3):
        """ Dual attention block using hyperparameters. """
        super().__init__()
        self.conv_head = nn.Sequential(
            Conv2d(n_channels, n_channels, kernel_size, padding_mode="reflect"),
            nn.LeakyReLU(),
            Conv2d(n_channels, n_channels, kernel_size, padding_mode="reflect", init_weights=init.xavier_normal_),
        )
        self.sp_attn = SpatialAttn(kernel_size)
        self.ch_attn = ChannelAttn(n_channels)
        self.conv_tail = HyperConv2d(h_embeddings, n_channels * 2, n_channels, 1, init_weights=init.xavier_normal_)

    def hypertrain(self, enable: bool):
        """ Enable/disable hyperparameter training. """
        for param in itertools.chain(self.conv_head.parameters(),
                                     self.sp_attn.parameters(),
                                     self.ch_attn.parameters()):
            param.requires_grad = not enable
        for param in self.conv_tail.parameters():
            param.requires_grad = enable

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        r = self.conv_head(x)
        sa = self.sp_attn(r)
        ca = self.ch_attn(r)
        r = torch.cat([sa, ca], dim=1)
        r = self.conv_tail(r, h)
        return x + r


class HyperResBlk(nn.Module):
    def __init__(self, h_embeddings: int, n_channels: int, n_dab: int = 2, kernel_size: int = 3):
        """ Recursive residual group using hyperparameters. """
        super().__init__()

        self.dabs = nn.ModuleList([
            HyperDualAttn(h_embeddings, n_channels, kernel_size)
            for _ in range(n_dab)
        ])
        self.conv_tail = HyperConv2d(h_embeddings, n_channels, n_channels, kernel_size,
                                     padding_mode="reflect", init_weights=init.xavier_normal_)

    def hypertrain(self, enable: bool):
        """ Enable/disable hyperparameter training. """
        for dab in self.dabs:
            dab: HyperDualAttn
            dab.hypertrain(enable)
        for param in self.conv_tail.parameters():
            param.requires_grad = enable

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = x
        for dab in self.dabs:
            r = dab(r, h)
        r = self.conv_tail(r, h)
        return x + r


class HyperPipeline(nn.Module):
    def __init__(self, h_embeddings: int, in_channels: int = 3, out_channels: int = 3,
                 mid_channels: int = 24, n_dab: int = 5, kernel_size: int = 3):
        """ pipeline using hyperparameters. """
        super().__init__()

        self.conv_head = HyperConv2d(h_embeddings,  in_channels, mid_channels, kernel_size,
                                     padding_mode="reflect", init_weights=init.xavier_normal_)
        self.conv_tail = HyperConv2d(h_embeddings, mid_channels, out_channels, kernel_size,
                                     padding_mode="reflect", init_weights=init.xavier_normal_)
        self.rrg1 = HyperResBlk(h_embeddings, mid_channels,     n_dab, kernel_size)
        self.rrg2 = HyperResBlk(h_embeddings, mid_channels * 4, n_dab, kernel_size)
        self.rrg3 = HyperResBlk(h_embeddings, mid_channels,     n_dab, kernel_size)
        self.downscale = nn.PixelUnshuffle(2)
        self.upscale   = nn.PixelShuffle(2)

    def hypertrain(self, enable: bool):
        """ Enable/disable hyperparameter training. """
        self.conv_head.hypertrain(enable)
        self.conv_tail.hypertrain(enable)
        self.rrg1.hypertrain(enable)
        self.rrg2.hypertrain(enable)
        self.rrg3.hypertrain(enable)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        y = x
        y = self.conv_head(y, h)
        y = self.rrg1(y, h)
        y = self.downscale(y)
        y = self.rrg2(y, h)
        y = self.upscale(y)
        y = self.rrg3(y, h)
        y = self.conv_tail(y, h)
        return y
