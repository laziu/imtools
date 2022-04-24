import torch
import kornia.color
import kornia.enhance
import numpy as np
from toolz import isiterable

from ..bayer.compat import pattern_to_mask
from ..convert import decompose_to_batch_tensor, recompose_from_batch_tensor
from ..misc import TArray

from . import batch as impl


def white_balance(x: TArray, wb: torch.Tensor, pattern: torch.Tensor = None) -> TArray:
    """ Apply white balance to the images.
    See `utils.color.torch.WhiteBalance` for details.

    Args:
        x: Input RGB image.
        wb: White balance factors for each instance.
        pattern: Bayer pattern for each instance if mode is `bayer`.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    wb: torch.Tensor = torch.as_tensor(wb).to(x.device)
    if wb.ndim <= 1:
        wb.unsqueeze(0)
    if wb.shape[0] == 1:
        wb = wb.repeat(x.shape[0], 1)

    if pattern is None:
        mode = "channel"
    else:
        mode = "bayer"
        bayer_mask = pattern_to_mask(pattern, x.shape).to(x.device)

    module = impl.WhiteBalance(mode)
    y = module(x, wb, bayer_mask)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def color_matrix(x: TArray, matrix: torch.Tensor) -> TArray:
    """ Apply color matrix to the images.

    Args:
        x: Input RGB image.
        matrix: Color matrix.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    matrix = torch.as_tensor(matrix).to(dtype=x.dtype, device=x.device)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).repeat(x.shape[0], 1, 1)
    elif matrix.ndim == 3:
        if matrix.shape[0] == 1:
            matrix = matrix.repeat(x.shape[0], 1, 1)
    else:
        raise ValueError(f"Invalid matrix shape: {matrix.shape}")

    module = impl.ColorMatrix()
    y = module(x, matrix)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def rgb2xyz(x: TArray) -> TArray:
    """ Convert RGB to XYZ. """
    x, *xinfo = decompose_to_batch_tensor(x)

    y = kornia.color.rgb_to_xyz(x)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def xyz2rgb(x: TArray) -> TArray:
    """ Convert XYZ to RGB. """
    x, *xinfo = decompose_to_batch_tensor(x)

    y = kornia.color.xyz_to_rgb(x)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def gamma(x: TArray, gamma: float) -> TArray:
    """ Apply gamma correction to the images. """
    x, *xinfo = decompose_to_batch_tensor(x)

    y = kornia.enhance.adjust_gamma(x, gamma)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def srgb2linear(x: TArray) -> TArray:
    """ Convert sRGB to linear RGB. """
    x, *xinfo = decompose_to_batch_tensor(x)

    y = kornia.color.rgb_to_linear_rgb(x)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def linear2srgb(x: TArray) -> TArray:
    """ Convert linear RGB to sRGB. """
    x, *xinfo = decompose_to_batch_tensor(x)

    y = kornia.color.linear_rgb_to_rgb(x)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y
