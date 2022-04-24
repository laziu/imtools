import torch
import numpy as np

from ..convert import decompose_to_batch_tensor, recompose_from_batch_tensor
from ..misc import TArray

from . import batch as impl


def pattern_to_mask(pattern: torch.Tensor, x_shape: tuple[int, int, int, int]) -> torch.Tensor:
    """ Convert Bayer pattern to Bayer mask. """
    B, C, H, W = x_shape

    pattern = torch.as_tensor(pattern)

    if pattern.ndim == 2:
        pattern = pattern.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    elif pattern.ndim == 3:
        pattern = pattern.unsqueeze(1)
        if pattern.shape[0] == 1:
            pattern = pattern.repeat(B, 1, 1, 1)
    elif pattern.ndim == 4:
        if pattern.shape[0] == 1:
            pattern = pattern.repeat(B, 1, 1, 1)

    assert pattern.shape[:2] == (B, 1), f"Invalid pattern shape {pattern.shape}"

    pattern = pattern.repeat(1, 1, int(np.ceil(H / 2)), int(np.ceil(W / 2)))[:, :, :H, :W]

    return pattern


def mosaic(x: TArray, pattern: torch.Tensor) -> TArray:
    """ Mosaicing RGB images to Bayer pattern.
    See `utils.bayer.torch.Mosaic` for details.

    Args:
        x: Input RGB image.
        pattern: Bayer pattern.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    bayer_mask = pattern_to_mask(pattern, x.shape).to(x.device)

    module = impl.Mosaic().to(x.device)
    y = module(x, bayer_mask)

    y = recompose_from_batch_tensor(y, *xinfo)

    return y


def demosaic3x3(x: TArray, pattern: torch.Tensor) -> TArray:
    """ Demosaicing of bayer images using bilinear interpolation.
    See `utils.bayer.torch.Demosaic3x3` for details.

    Args:
        x: Input bayer image.
        pattern: Bayer pattern.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    bayer_mask = pattern_to_mask(pattern, x.shape).to(x.device)

    module = impl.Demosaic3x3().to(x.device)
    y = module(x, bayer_mask)

    y = recompose_from_batch_tensor(y, *xinfo)
    return y


def demosaic5x5(x: TArray, pattern: torch.Tensor) -> TArray:
    """ Demosaicing of bayer images using Malvar-He-Cutler method.
    See `utils.bayer.torch.Demosaic5x5` for details.

    Args:
        x: Input bayer image.
        pattern: Bayer pattern.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    bayer_mask = pattern_to_mask(pattern, x.shape).to(x.device)

    module = impl.Demosaic5x5().to(x.device)
    y = module(x, bayer_mask)

    y = recompose_from_batch_tensor(y, *xinfo)
    return y


def demosaic_ahd(x: TArray, pattern: torch.Tensor) -> TArray:
    """ Demosaicing of bayer images Adaptive Homogeneity-Directed method.
    See `utils.bayer.torch.DemosaicAHD` for details.

    Args:
        x: Input bayer image.
        pattern: Bayer pattern.
    """
    x, *xinfo = decompose_to_batch_tensor(x)

    bayer_mask = pattern_to_mask(pattern, x.shape).to(x.device)

    module = impl.DemosaicAHD().to(x.device)
    y = module(x, bayer_mask)

    y = recompose_from_batch_tensor(y, *xinfo)
    return y
