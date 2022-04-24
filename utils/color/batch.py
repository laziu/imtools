from typing import Literal

import torch
import torch.nn as nn
import kornia.color
import kornia.enhance
import numpy as np


class WhiteBalance(nn.Module):
    def __init__(self, mode: Literal["channel", "bayer"] = "bayer"):
        """ Apply white balance to the images.

        Args:
            mode: Mode of the white balance. Available options:
                - `channel`: Apply for multichannel images, such as RGB image.
                - `bayer`: Apply for Bayer images. `pattern` must be provided.
        """
        super().__init__()
        self.mode = mode

    def __call__(self, x: torch.Tensor, wb: torch.Tensor, bayer_mask: torch.Tensor = None) -> torch.Tensor:
        """ Mosaicing RGB images to Bayer pattern.

        Args:
            x: Input RGB image.
            wb: White balance factors for each instance.
            bayer_mask: Bayer pattern for each instance if mode is `bayer`.
        """
        return super().__call__(x, wb, bayer_mask)

    def forward(self, x: torch.Tensor, wb: torch.Tensor, bayer_mask: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        C = 3 if self.mode == "bayer" else C

        wb = wb.to(x.dtype).view(B, C, 1, 1)

        if self.mode == "bayer":
            x_rgb = torch.cat([x, x, x], dim=1) * wb

            assert bayer_mask.shape[-2:] == (H, W), "bayer_mask shape must be (B, 1, H, W)"

            x = torch.gather(x_rgb, dim=1, index=bayer_mask)
        else:
            x = x * wb

        return x


class ColorMatrix(nn.Module):
    def __init__(self, matrix: torch.Tensor = None):
        """ Apply color matrix to the images.

        Args:
            matrix: Color matrix, set to `None` to use individual matrix for each instance.
        """
        super().__init__()
        self.matrix = torch.as_tensor(matrix) if matrix else None

    def _apply(self, fn):
        if self.matrix is not None:
            self.matrix = fn(self.matrix)
        return super()._apply(fn)

    def __call__(self, x: torch.Tensor, matrix: torch.Tensor = None) -> torch.Tensor:
        """ Apply color matrix to the images.

        Args:
            x: Input image.
            matrix: Color matrix for each instance which overrides the default matrix.
        """
        return super().__call__(x, matrix)

    def forward(self, x: torch.Tensor, matrix: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(-2, -1)  # B, H*W, C

        matrix = matrix.view(B, C, C) if matrix is not None else self.matrix.repeat(B, 1, 1)

        assert C == matrix.size(-1), "Color matrix can only be applied to RGB images"
        x = torch.bmm(x, matrix.transpose(-1, -2))  # B, H*W, C

        x = x.swapaxes(-1, -2).view(B, C, H, W)
        return x


class RGB2XYZ(nn.Module):
    def __init__(self) -> None:
        """ Convert RGB to XYZ. """
        super().__init__()
        self.module = kornia.color.RgbToXyz()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert RGB to XYZ.

        Args:
            x: Input image.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class XYZ2RGB(nn.Module):
    def __init__(self) -> None:
        """ Convert XYZ to RGB. """
        super().__init__()
        self.module = kornia.color.XyzToRgb()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert XYZ to RGB.

        Args:
            x: Input image.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Gamma(nn.Module):
    def __init__(self, gamma: float):
        """ Adjust gamma of an image.

        Args:
            gamma: Gamma value.
        """
        super().__init__()
        self.module = kornia.enhance.AdjustGamma(gamma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Adjust gamma of an image.

        Args:
            x: Input image.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class SRGB2Linear(nn.Module):
    def __init__(self) -> None:
        """ Convert sRGB to linear RGB. """
        super().__init__()
        self.module = kornia.color.RgbToLinearRgb()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert sRGB to linear RGB.

        Args:
            x: Input image.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Linear2SRGB(nn.Module):
    def __init__(self):
        """ Convert linear RGB to sRGB. """
        super().__init__()
        self.module = kornia.color.LinearRgbToRgb()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert linear RGB to sRGB.

        Args:
            x: Input image.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Brightness(nn.Module):
    def __init__(self, multiplier: float):
        """ Adjust brightness of an image.

        Args:
            img: Image to be adjusted.
            multiplier: Brightness multiplier, i.e. 0=black image, 1=original image.
        """
        super().__init__()


class Contrast(nn.Module):
    def __init__(self, multiplier: float):
        """ Adjust contrast of an image.

        Args:
            img: Image to be adjusted.
            multiplier: Contrast multiplier, i.e. 0=solid gray image, 1=original image.
        """


class Saturation(nn.Module):
    def __init__(self, multiplier: float):
        """ Adjust saturation of an image.

        Args:
            img: Image to be adjusted.
            multiplier: Saturation multiplier, i.e. 0=grayscale image, 1=original image.
        """
        super().__init__()


class Hue(nn.Module):
    def __init__(self, angle: float):
        """ Adjust hue of an image.

        Args:
            img: Image to be adjusted.
            angle: Hue angle in [-0.5, +0.5], i.e. 0=original image.
        """
        super().__init__()


class ToGrayscale(nn.Module):
    def __init__(self, num_output_channels: int = 1) -> None:
        """ Convert image to grayscale, see `to_grayscale` for more details. """
        super().__init__()
        self.num_output_channels = num_output_channels


class Invert(nn.Module):
    """ Invert the image, see `invert` for more details. """


class Posterize(nn.Module):
    def __init__(self, bits: int) -> None:
        """ Reduce the number of bits for each pixel, see `posterize` for more details. """
        super().__init__()
        self.bits = bits


class Equalize(nn.Module):
    """ Equalize the image histogram, see `equalize` for more details. """
