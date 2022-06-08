from typing import Callable
from collections import OrderedDict
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


class HyperConv2dKernelNet(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, kernel_size: int, **kwargs):
        """ Embed hyperparameters to a convolution kernel.

        Args:
            h_elements: number of elements of the hyperparameter vector.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the convolution kernel.
        """
        super().__init__()

        mid1_channels = 16
        mid2_channels = 32
        mid3_channels = 32
        self.fc = nn.Linear(h_elements, mid1_channels * kernel_size ** 2)
        self.resize = Reshape((-1, mid1_channels, kernel_size, kernel_size))
        self.conv1 = nn.Conv2d(mid1_channels, mid2_channels, 1)
        self.conv2 = nn.Conv2d(mid2_channels, mid3_channels, 1)
        self.conv3 = nn.Conv2d(mid3_channels, n_channels, 1)
        self.relu = nn.LeakyReLU()

        init.kaiming_normal_(self.fc.weight)
        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)
        init.xavier_normal_(self.conv3.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        k = self.relu(self.fc(h))
        k = self.resize(k)
        k = self.relu(self.conv1(k))
        k = self.relu(self.conv2(k))
        k = self.conv3(k)
        return k


class HyperConv2d(nn.Module):
    def __init__(self, h_elements: int, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros", **kwargs):
        """ Hyper-parameter embedding. """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.kernel_extention = HyperConv2dKernelNet(h_elements, out_channels * (in_channels // groups),
                                                     kernel_size, **kwargs)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **kwargs))
            fan_in = in_channels * kernel_size ** 2
            bound = 1 / (fan_in ** 0.5)
            init.uniform_(self.bias, -bound, +bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel_extention(h)

        y = []
        for b in range(x.size(0)):
            weight = kernels[b].view(self.out_channels, self.in_channels / self.groups,
                                     self.kernel_size, self.kernel_size)
            xb = x[b:b + 1]

            if self.padding_mode != "zeros":
                pad = (self.padding, self.padding, self.padding, self.padding)
                xb = F.conv2d(F.pad(xb, pad, mode=self.padding_mode), weight, self.bias,
                              self.stride, 0, self.dilation, self.groups)
            else:
                xb = F.conv2d(xb, weight, self.bias,
                              self.stride, self.padding, self.dilation, self.groups)

            y.append(xb)

        return torch.cat(y, dim=0)


class PseudoHyperConv2d(HyperConv2d):
    def __init__(self, h_elements: int, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros", **kwargs):
        """ Hyper-parameter embedding. """
        super().__init__(h_elements, in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups,
                         bias, padding_mode, **kwargs)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups,
                                               kernel_size, kernel_size), **kwargs)

        init.kaiming_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel_extention(h)

        y = []
        for b in range(x.size(0)):
            weight = kernels[b].view(self.out_channels, self.in_channels // self.groups,
                                     self.kernel_size, self.kernel_size)
            weight = weight + self.weight
            xb = x[b:b + 1]

            if self.padding_mode != "zeros":
                pad = (self.padding, self.padding, self.padding, self.padding)
                xb = F.conv2d(F.pad(xb, pad, mode=self.padding_mode), weight, self.bias,
                              self.stride, 0, self.dilation, self.groups)
            else:
                xb = F.conv2d(xb, weight, self.bias,
                              self.stride, self.padding, self.dilation, self.groups)

            y.append(xb)

        return torch.cat(y, dim=0)


class ChannelAttention(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 2):
        """ Channel attention layer using hyperparameters. """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1   = nn.Conv2d(n_channels, n_channels // reduction, 1)
        self.relu    = nn.PReLU(n_channels // reduction, 0.1)
        self.conv2   = nn.Conv2d(n_channels // reduction, n_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        s = self.avgpool(x)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        """ Spatial attention layer using hyperparameters. """
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        avgpool = torch.mean(x, dim=1, keepdim=True)
        maxpool = torch.max(x,  dim=1, keepdim=True)[0]
        s = torch.cat([avgpool, maxpool], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


class HyperDualAttention(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, kernel_size: int = 3, reduction: int = 2):
        """ Dual attention block using hyperparameters. """
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.relu  = nn.PReLU(n_channels, 0.1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.sp_attn = SpatialAttention(kernel_size)
        self.ch_attn = ChannelAttention(n_channels, reduction)
        self.conv_tail = HyperConv2d(h_elements, n_channels * 2, n_channels, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)
        sa = self.sp_attn(r)
        ca = self.ch_attn(r)
        r = torch.cat([sa, ca], dim=1)
        r = self.conv_tail(r, h)
        return x + r


class PseudoHyperDualAttention(HyperDualAttention):
    def __init__(self, h_elements: int, n_channels: int, kernel_size: int = 3, reduction: int = 2):
        """ Dual attention block using hyperparameters. """
        super().__init__(h_elements, n_channels, kernel_size, reduction)
        self.conv_tail = PseudoHyperConv2d(h_elements, n_channels * 2, n_channels, 1)


class HyperResidualBlock(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, n_dab: int = 2, kernel_size: int = 3, reduction: int = 2):
        """ Recursive residual group using hyperparameters. """
        super().__init__()
        self.set_modules(h_elements, n_channels, n_dab, kernel_size, reduction)

    def set_modules(self, h_elements, n_channels, n_dab, kernel_size, reduction):
        self.dabs = nn.ModuleList([
            HyperDualAttention(h_elements, n_channels, kernel_size, reduction)
            for i in range(n_dab)
        ])
        self.conv_tail = HyperConv2d(h_elements, n_channels, n_channels, kernel_size,
                                     padding=(kernel_size - 1) // 2, padding_mode="reflect")

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = x
        for d in self.dabs:
            r = d(r, h)
        r = self.conv_tail(r, h)
        return x + r


class PseudoHyperResidualBlock(HyperResidualBlock):
    def set_modules(self, h_elements, n_channels, n_dab, kernel_size, reduction):
        self.dabs = nn.ModuleList([
            PseudoHyperDualAttention(h_elements, n_channels, kernel_size, reduction)
            for i in range(n_dab)
        ])
        self.conv_tail = PseudoHyperConv2d(h_elements, n_channels, n_channels, kernel_size,
                                           padding=(kernel_size - 1) // 2, padding_mode="reflect")


class HyperPipeline(nn.Module):
    def __init__(self, h_elements: int = 16, in_channels: int = 3, out_channels: int = 3,
                 mid_channels: int = 24, n_dab: int = 5, kernel_size: int = 3):
        """ Basic Pipeline for RGB2RAW. """
        super().__init__()
        self.set_modules(h_elements, in_channels, out_channels, mid_channels, n_dab, kernel_size)

    def set_modules(self, h_elements, in_channels, out_channels, mid_channels, n_dab, kernel_size):
        self.conv_head = HyperConv2d(h_elements, in_channels,  mid_channels, kernel_size,
                                     padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.conv_tail = HyperConv2d(h_elements, mid_channels, out_channels, kernel_size,
                                     padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.rrg1 = HyperResidualBlock(h_elements, mid_channels,     n_dab, kernel_size)
        self.rrg2 = HyperResidualBlock(h_elements, mid_channels * 4, n_dab, kernel_size)
        self.rrg3 = HyperResidualBlock(h_elements, mid_channels,     n_dab, kernel_size)
        self.downscale = nn.PixelUnshuffle(2)
        self.upscale   = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        x = self.conv_head(x, h)
        x = self.rrg1(x, h)
        x = self.downscale(x)
        x = self.rrg2(x, h)
        x = self.upscale(x)
        x = self.rrg3(x, h)
        x = self.conv_tail(x, h)
        return x


class PseudoHyperPipeline(HyperPipeline):
    def set_modules(self, h_elements, in_channels, out_channels, mid_channels, n_dab, kernel_size):
        self.conv_head = PseudoHyperConv2d(h_elements, in_channels,  mid_channels, kernel_size,
                                           padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.conv_tail = PseudoHyperConv2d(h_elements, mid_channels, out_channels, kernel_size,
                                           padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.rrg1 = PseudoHyperResidualBlock(h_elements, mid_channels,     n_dab, kernel_size)
        self.rrg2 = PseudoHyperResidualBlock(h_elements, mid_channels * 4, n_dab, kernel_size)
        self.rrg3 = PseudoHyperResidualBlock(h_elements, mid_channels,     n_dab, kernel_size)
        self.downscale = nn.PixelUnshuffle(2)
        self.upscale   = nn.PixelShuffle(2)
