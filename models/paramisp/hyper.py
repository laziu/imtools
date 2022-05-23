from typing import Callable
from collections import OrderedDict
import time

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
    def __init__(self, h_elements: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        """ Embed hyperparameters to a convolution kernel.

        Args:
            h_elements: number of elements of the hyperparameter vector.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the convolution kernel.
        """
        super().__init__()

        n_channels = out_channels * in_channels * kernel_size ** 2

        ch1 = n_channels
        ch2 = n_channels // 2
        ch3 = n_channels // 4
        ch4 = n_channels // 8

        self.fc1 = nn.Linear(h_elements, ch1)
        self.fc2 = nn.Linear(h_elements + ch1, ch2)
        self.fc3 = nn.Linear(h_elements + ch1 + ch2, ch3)
        self.fc4 = nn.Linear(h_elements + ch1 + ch2 + ch3, ch4)
        self.fc5 = nn.Linear(h_elements + ch1 + ch2 + ch3 + ch4, n_channels)

        self.relu = nn.LeakyReLU()
        self.resize = Reshape((-1, out_channels, in_channels, kernel_size, kernel_size))

        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc2.weight)
        init.kaiming_normal_(self.fc3.weight)
        init.kaiming_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        fc1 = self.relu(self.fc1(h))
        fc2 = self.relu(self.fc2(torch.cat((h, fc1), dim=0)))
        fc3 = self.relu(self.fc3(torch.cat((h, fc1, fc2), dim=0)))
        fc4 = self.relu(self.fc4(torch.cat((h, fc1, fc2, fc3), dim=0)))
        fc5 = self.fc5(torch.cat((h, fc1, fc2, fc3, fc4), dim=0))
        return self.resize(fc5)


class HyperConv2d(nn.Module):
    def __init__(self, h_elements: int, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros", **kwargs):
        """ Hyper-parameter embedding. """
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.kernel_extention = HyperConv2dKernelNet(h_elements, in_channels, out_channels, kernel_size, **kwargs)

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
            weight = kernels[b]
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


class HyperChannelAttention(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, reduction: int = 2):
        """ Channel attention layer using hyperparameters. """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1   = HyperConv2d(h_elements, n_channels, n_channels // reduction, 1)
        self.relu    = nn.PReLU(n_channels // reduction, 0.1)
        self.conv2   = HyperConv2d(h_elements, n_channels // reduction, n_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        s = self.avgpool(x)
        s = self.conv1(s, h)
        s = self.relu(s)
        s = self.conv2(s, h)
        s = self.sigmoid(s)
        return x * s


class HyperSpatialAttention(nn.Module):
    def __init__(self, h_elements: int, kernel_size: int = 3):
        """ Spatial attention layer using hyperparameters. """
        super().__init__()
        self.conv    = HyperConv2d(h_elements, 2, 1, kernel_size,
                                   padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        avgpool = torch.mean(x, dim=1, keepdim=True)
        maxpool = torch.max(x,  dim=1, keepdim=True)[0]
        s = torch.cat([avgpool, maxpool], dim=1)
        s = self.conv(s, h)
        s = self.sigmoid(s)
        return x * s


class HyperDualAttention(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, kernel_size: int = 3, reduction: int = 2):
        """ Dual attention block using hyperparameters. """
        super().__init__()
        self.conv1 = HyperConv2d(h_elements, n_channels, n_channels, kernel_size,
                                 padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.relu  = nn.PReLU(n_channels, 0.1)
        self.conv2 = HyperConv2d(h_elements, n_channels, n_channels, kernel_size,
                                 padding=(kernel_size - 1) // 2, padding_mode="reflect")
        self.sp_attn = HyperSpatialAttention(h_elements, kernel_size)
        self.ch_attn = HyperChannelAttention(h_elements, n_channels, reduction)
        self.conv_tail = HyperConv2d(h_elements, n_channels * 2, n_channels, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = self.conv1(x, h)
        r = self.relu(r)
        r = self.conv2(r, h)
        sa = self.sp_attn(r, h)
        ca = self.ch_attn(r, h)
        r = torch.cat([sa, ca], dim=1)
        r = self.conv_tail(r, h)
        return x + r


class HyperResidualBlock(nn.Module):
    def __init__(self, h_elements: int, n_channels: int, n_dab: int = 2, kernel_size: int = 3, reduction: int = 2):
        """ Recursive residual group using hyperparameters. """
        super().__init__()
        self.dabs = nn.Sequential(OrderedDict([
            (f"dab{i+1}", HyperDualAttention(h_elements, n_channels, kernel_size, reduction))
            for i in range(n_dab)
        ]))
        self.conv_tail = HyperConv2d(h_elements, n_channels, n_channels, kernel_size,
                                     padding=(kernel_size - 1) // 2, padding_mode="reflect")

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = self.dabs(x, h)
        r = self.conv_tail(r, h)
        return x + r


class HyperPipeline(nn.Module):
    def __init__(self, h_elements: int = 16, in_channels: int = 3, out_channels: int = 3,
                 mid_channels: int = 24, n_dab: int = 5, kernel_size: int = 3):
        """ Basic Pipeline for RGB2RAW. """
        super().__init__()

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
