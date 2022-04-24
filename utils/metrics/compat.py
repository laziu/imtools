from torch import Tensor
import numpy as np
import skimage.metrics

from ..convert import reshape_numpy, reshape_tensor, im2tensor
from ..misc import TArray

from . import torch as ptloss


def ssim_loss(x: TArray, y: TArray) -> TArray:
    """ Compute the Structrual dissimilarity (DSSIM), i.e. SSIM loss.

    Args:
        x: Tensor of shape (B, C, H, W)
        y: Tensor of shape (B, C, H, W)
        window_size: Size of the window to use.
    """
    if isinstance(x, Tensor):
        x = reshape_tensor(x, batch=True)
        y = reshape_tensor(y, batch=True)
        return ptloss.ssim_loss(x, y)
    else:
        x = reshape_numpy(x)
        y = reshape_numpy(y)
        return 0.5 * (1 - skimage.metrics.structural_similarity(x, y, channel_axis=-1))


def ssim(x: TArray, y: TArray) -> TArray:
    """ Compute the Structural similarity (SSIM).

    Args:
        x: input image
        y: label image
        window_size: Size of the window to use.
    """
    return 1 - 2 * ssim_loss(x, y)


def psnr_loss(x: TArray, y: TArray) -> TArray:
    """ Compute the negative Peak Signal-to-Noise Ratio (-PSNR), i.e. PSNR loss. """
    if isinstance(x, Tensor):
        x = reshape_tensor(x, batch=True)
        y = reshape_tensor(y, batch=True)
        return ptloss.psnr_loss(x, y)
    else:
        x = reshape_numpy(x)
        y = reshape_numpy(y)
        return -skimage.metrics.peak_signal_noise_ratio(x, y)


def psnr(x: TArray, y: TArray) -> TArray:
    """ Compute the Peak Signal-to-Noise Ratio (PSNR).

    Args:
        x: input image
        y: label image
    """
    return -psnr_loss(x, y)


def vgg_loss(x: TArray, y: TArray) -> TArray:
    """ Functional version of `VGGLoss`. """
    x = im2tensor(x, batch=True)
    y = im2tensor(y, batch=True)
    return ptloss.vgg_loss(x, y)
