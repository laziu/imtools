import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class HyperConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, h_length: int, bias=True):
        """ Hyper-parameter embedding. """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size

        n_params1 = h_length * 4
        n_params2 = (out_channels * in_channels) // 4 * kernel_size ** 2

        self.get_weight = nn.Sequential(
            init.kaiming_normal(nn.Linear(h_length, n_params1)),
            nn.PReLU(n_params1),
            init.kaiming_normal(nn.Linear(n_params1, n_params2)),
            nn.PReLU(n_params2),
            init.xavier_normal(nn.Linear(n_params2, in_channels * out_channels * kernel_size ** 2)),
            Reshape((in_channels * out_channels, kernel_size, kernel_size)),
            init.kaiming_normal(nn.Conv2d(in_channels * out_channels, in_channels * out_channels, 1)),
            nn.PReLU(in_channels * out_channels),
            init.kaiming_normal(nn.Conv2d(in_channels * out_channels, in_channels * out_channels, 1)),
            Reshape((in_channels, out_channels, kernel_size, kernel_size)),
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight(h)
        if self.kernel_size > 1:
            pad = (self.kernel_size - 1) // 2
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, weight, self.bias)
        return x
