from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .layers import DualAttentionBlock, RecursiveResidualGroup


class BasicPipeline(nn.Module):
    def __init__(self, n_rrg: int = 3, n_dab: int = 5, n_feat: int = 24, reduction: int = 4, ksize=3):
        """ Basic Pipeline for RGB2RAW. """
        super().__init__()
        self.conv_head = nn.Conv2d(3, n_feat, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        self.rrgs = nn.Sequential(OrderedDict([
            (f"rrg{i+1}", RecursiveResidualGroup(n_feat, n_dab, ksize, reduction))
            for i in range(n_rrg)
        ]))
        self.conv_tail = nn.Conv2d(n_feat, 3, ksize, padding=(ksize - 1) // 2, padding_mode="reflect")
        init.xavier_normal_(self.conv_head.weight)
        init.xavier_normal_(self.conv_tail.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        y = x
        y = self.conv_head(y)
        y = self.rrgs(y)
        y = self.conv_tail(y)
        return y
