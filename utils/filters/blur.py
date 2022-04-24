import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.filters
import skimage.filters

from ..convert import reshape_tensor
from ..misc import TArray, ensure_tuple


class Gaussian(nn.Module):
    def __init__(self, sigma: float | tuple[float, float]):
        """ Module version of `gaussian`. """
        super().__init__()
        sigma = np.asarray(ensure_tuple(sigma, 2))
        ksize = [int(ksz) for ksz in 1 + 2 * np.ceil(3 * sigma).astype(int)]
        self.blur = kornia.filters.GaussianBlur2d(kernel_size=ksize, sigma=sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.blur(batch)


def gaussian(image: TArray, sigma: float | tuple[float, float]) -> TArray:
    """ Apply gaussian filter to image.

    Args:
        image: input image.
        sigma: stdev for kernel.
    """
    if isinstance(image, torch.Tensor):
        image = reshape_tensor(image)

        sigma = np.asarray(ensure_tuple(sigma, 2))
        ksize = [int(ksz) for ksz in 1 + 2 * np.ceil(3 * sigma).astype(int)]

        return kornia.filters.gaussian_blur2d(image, kernel_size=ksize, sigma=sigma)
    else:
        return skimage.filters.gaussian(image, sigma=sigma, channel_axis=-1)
