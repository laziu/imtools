"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init


class ChannelAttention(nn.Module):
    def __init__(self, n_channel: int, reduction: int = 2):
        """ Channel attention layer. """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1   = nn.Conv2d(n_channel, n_channel // reduction, 1)
        self.relu    = nn.PReLU(n_channel // reduction, 0.1)
        self.conv2   = nn.Conv2d(n_channel // reduction, n_channel, 1)
        self.sigmoid = nn.Sigmoid()
        # init.kaiming_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        s = self.avgpool(x)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.sigmoid(s)
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, ksize: int = 3):
        """ Spatial attention layer. """
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()
        init.xavier_normal_(self.conv.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        avgpool = torch.mean(x, dim=1, keepdim=True)
        maxpool = torch.max(x,  dim=1, keepdim=True)[0]
        s = torch.cat([avgpool, maxpool], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


class DualAttentionBlock(nn.Module):
    def __init__(self, n_feat: int, ksize: int = 3, reduction: int = 2):
        """ Dual attention block. """
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.relu  = nn.PReLU(n_feat, 0.1)
        self.conv2 = nn.Conv2d(n_feat, n_feat, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.sp_attn = SpatialAttention(ksize)
        self.ch_attn = ChannelAttention(n_feat, reduction)
        self.conv_tail = nn.Conv2d(n_feat * 2, n_feat, 1)
        # init.kaiming_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)
        init.xavier_normal_(self.conv_tail.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)
        sa = self.sp_attn(r)
        ca = self.ch_attn(r)
        r = torch.cat([sa, ca], dim=1)
        r = self.conv_tail(r)
        return x + r


class RecursiveResidualGroup(nn.Module):
    def __init__(self, n_feat: int, n_dab: int = 2, ksize: int = 3, reduction: int = 2):
        """ Recursive residual group. """
        super().__init__()
        self.dabs = nn.Sequential(OrderedDict([
            (f"dab{i+1}", DualAttentionBlock(n_feat, ksize, reduction))
            for i in range(n_dab)
        ]))
        self.conv_tail = nn.Conv2d(n_feat, n_feat, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        # init.kaiming_normal_(self.conv_tail.weight)
        init.xavier_normal_(self.conv_tail.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        r = self.dabs(x)
        r = self.conv_tail(r)
        return x + r
