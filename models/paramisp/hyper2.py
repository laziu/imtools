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


class DenseLinear(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.fc1 = Linear(n_channels, n_channels)
        self.fc2 = Linear(n_channels * 2, n_channels)
        self.fc3 = Linear(n_channels * 3, n_channels)
        self.fc4 = Linear(n_channels * 4, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = F.leaky_relu(self.fc1(x))
        y2 = F.leaky_relu(self.fc2(torch.cat([x, y1], dim=-1)))
        y3 = F.leaky_relu(self.fc3(torch.cat([x, y1, y2], dim=-1)))
        y4 = F.leaky_relu(self.fc4(torch.cat([x, y1, y2, y3], dim=-1)))
        return y4


class HyperConv2dKernelNet(nn.Module):
    def __init__(self, h_embeddings: int, in_channels: int, out_channels: int, kernel_size: int,
                 *, mid_channels: int = 256, n_blocks: int = 1):
        """ Embed hyperparameters to a convolution kernel. """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # conv weight: (out_channels, in_channels, kernel_size, kernel_size)
        # conv bias:   (out_channels)

        def init_decomposed_xavier_normal_(w: torch.Tensor, gain: float = 0.03):
            std_target = math.sqrt(2.0 / ((out_channels + in_channels) * (kernel_size ** 2)))
            std_vector = std_target ** (1 / 3)
            with torch.no_grad():
                w.normal_(0, gain * std_vector)

        layers = []
        layers.append(Linear(h_embeddings, mid_channels))
        layers.append(nn.LeakyReLU())
        for _ in range(n_blocks):
            layers.append(DenseLinear(mid_channels))
            layers.append(nn.LeakyReLU())
        layers.append(Linear(mid_channels, in_channels + out_channels + kernel_size ** 2,
                             init_weights=init_decomposed_xavier_normal_))
        self.layers = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weight: torch.Tensor = self.layers(h)

        cut1 = self.out_channels
        cut2 = cut1 + self.in_channels

        gamma, phi, psi = weight[:cut1], weight[cut1:cut2], weight[cut2:]
        return (gamma.view(self.out_channels, 1, 1, 1) *
                phi.view(1,    self.in_channels, 1, 1) *
                psi.view(1, 1, self.kernel_size, self.kernel_size))

        cut = self.in_channels + self.out_channels + self.kernel_size ** 2
        alpha = self.compose_kernel(weights[..., :cut]) * 1.0
        beta  = self.compose_kernel(weights[..., cut:]) * 0.05

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
    def __init__(self, h_embeddings: int, in_channels: int, out_channels: int, kernel_size: int,
                 gain_alpha: float = 1.0, gain_beta: float = 0.05, ** kwargs):
        """ Embed hyperparameters to a convolution. """
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.gain_alpha = gain_alpha
        self.gain_beta  = gain_beta
        self.alpha = HyperConv2dKernelNet(h_embeddings, in_channels, out_channels, kernel_size)
        self.beta  = HyperConv2dKernelNet(h_embeddings, in_channels, out_channels, kernel_size)

    def hypertrain(self, enable: bool):
        """ Enable/disable hyperparameter training. """
        for param in itertools.chain(self.alpha.parameters(),
                                     self.beta.parameters()):
            param.requires_grad = enable
        self.weight.requires_grad = not enable
        self.bias.requires_grad   = not enable

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            return self._conv_forward(x, self.weight, self.bias)
        else:
            def calculate_weight(hparam: torch.Tensor) -> torch.Tensor:
                alpha: torch.Tensor = self.alpha(hparam) * self.gain_alpha
                beta:  torch.Tensor = self.beta(hparam)  * self.gain_beta
                return (1 + F.elu(alpha)) * self.weight + beta

            inputs  = x.reshape(1, -1, x.size(2), x.size(3))
            weights = torch.cat([calculate_weight(hparam) for hparam in h], dim=0)

            y: torch.Tensor
            if self.padding_mode != "zeros":
                y = F.conv2d(F.pad(inputs, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                             weights, stride=self.stride,
                             padding=0, dilation=self.dilation, groups=self.groups * x.size(0))
            else:
                y = F.conv2d(inputs, weights, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, groups=self.groups * x.size(0))

            y = y.reshape(x.size(0), -1, y.size(2), y.size(3))
            y = y + self.bias.view(1, -1, 1, 1)
            return y


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
        self.conv_tail.hypertrain(enable)

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
        self.conv_tail.hypertrain(enable)

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
