import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color
import numpy as np


class Mosaic(nn.Module):
    def __init__(self):
        """ Mosaicing RGB images to Bayer pattern. """
        super().__init__()

    def __call__(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        """ Mosaicing RGB images to Bayer pattern.

        Args:
            x: Input RGB image.
            bayer_mask: bayer pattern of the input images in Bx1xHxW format.
        """
        return super().__call__(x, bayer_mask)

    def forward(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert bayer_mask.shape[-2] == H and bayer_mask.shape[-1] == W, "Invalid mask shape"
        return torch.gather(x, dim=1, index=bayer_mask).view(B, 1, H, W)


class Demosaic3x3(nn.Module):
    def __init__(self):
        """ Demosaicing of bayer images using bilinear interpolation.

        References: https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py
        """
        super().__init__()

        self.kernels = self._get_interpolation_kernels()
        self.indices = self._get_gathering_indices()

    @classmethod
    def _get_interpolation_kernels(cls):
        return torch.tensor([
            # +
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]],
            # x
            [[1, 0, 1],
             [0, 0, 0],
             [1, 0, 1]],
            # -
            [[0, 0, 0],
             [2, 0, 2],
             [0, 0, 0]],
            # |
            [[0, 2, 0],
             [0, 0, 0],
             [0, 2, 0]],
        ], dtype=torch.get_default_dtype()).view(4, 1, 3, 3) / 4.0

    @classmethod
    def _get_gathering_indices(cls):
        index_rggb = torch.tensor([
            # R
            [4, 2],  # R G
            [3, 1],  # G B
            # G
            [0, 4],  # R G
            [4, 0],  # G B
            # B
            [1, 3],  # R G
            [2, 4],  # G B
        ]).view(1, 3, 2, 2)

        return {
            0x0112: index_rggb,
            0x1021: index_rggb.roll(1, dims=-1),
            0x1201: index_rggb.roll(1, dims=-2),
            0x2110: index_rggb.roll(1, dims=-1).roll(1, dims=-2),
        }

    @classmethod
    def _pattern_to_code(cls, pattern: torch.Tensor) -> int:
        return int((pattern[0] * 0x1000 + pattern[1] * 0x100 + pattern[2] * 0x10 + pattern[3]).item())

    def _apply(self, fn):
        self.kernels = fn(self.kernels)
        self.indices = {k: fn(v) for k, v in self.indices.items()}
        return super()._apply(fn)

    def __call__(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        """ Demosaicing of bayer images using bilinear interpolation.

        Args:
            x: Input bayer images.
            bayer_mask: bayer pattern of the input images in Bx1xHxW format.

        Returns:
            torch.Tensor: Demosaiced images.
        """
        return super().__call__(x, bayer_mask)

    def forward(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feats = self._calc_interpolated_features(x)
        indices = self._calc_required_indices(x, bayer_mask)
        return torch.gather(feats, dim=1, index=indices).view(B, 3, H, W)

    def _calc_interpolated_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 1, "Input image must be grayscale"

        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        feats = F.conv2d(x_pad, self.kernels.to(x.device))
        feats = torch.cat((feats, x), dim=1)  # B, 5, H, W
        return feats

    def _calc_required_indices(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        pattern = [self._pattern_to_code(p.flatten()) for p in bayer_mask[:, :, :2, :2]]
        indices = torch.cat([self.indices[p] for p in pattern], dim=0)  # B, 3, 2, 2

        halfH = int(np.ceil(H / 2))
        halfW = int(np.ceil(W / 2))
        indices = indices.to(x.device).repeat(1, 1, halfH, halfW)[:, :, :H, :W]  # B, 3, H, W

        return indices


class Demosaic5x5(Demosaic3x3):
    def __init__(self):
        """ Apply demosaicing to the input image using Malvar-He-Cutler method.

        References:
            - Malvar, Henrique S., Li-wei He, and Ross Cutler.
            "High-quality linear interpolation for demosaicing of Bayer-patterned color images."
            2004 IEEE International Conference on Acoustics, Speech, and Signal Processing. Vol. 3. IEEE, 2004.

        Args:
            pattern: Bayer pattern of the input image, e.g. `"RGGB"` or `Pattern.GRBG`, etc.
            pattern_per_instance: If True, `forward` takes the pattern for each instance of the image.
        """
        super().__init__()

    def __call__(self, x: torch.Tensor, bayer_mask: torch.Tensor) -> torch.Tensor:
        """ Demosaicing of bayer images using Malvar-He-Cutler method.

        Args:
            x: Input bayer images.
            pattern_per_instance: pattern per each instance in input batch.

        Returns:
            torch.Tensor: Demosaiced images.
        """
        return super().__call__(x, bayer_mask)

    @classmethod
    def _get_interpolation_kernels(cls):
        return torch.tensor([
            # + (G at R,B locations)
            [+0,  0, -2,  0,  0],
            [+0,  0,  4,  0,  0],
            [-2,  4,  8,  4, -2],
            [+0,  0,  4,  0,  0],
            [+0,  0, -2,  0,  0],
            # x (R at B and B at R)
            [+0,  0, -3,  0,  0],
            [+0,  4,  0,  4,  0],
            [-3,  0, 12,  0, -3],
            [+0,  4,  0,  4,  0],
            [+0,  0, -3,  0,  0],
            # - (R,B at G in R rows)
            [+0,  0,  1,  0,  0],
            [+0, -2,  0, -2,  0],
            [-2,  8, 10,  8, -2],
            [+0, -2,  0, -2,  0],
            [+0,  0,  1,  0,  0],
            # | (R,B at G in B rows)
            [0,  0, -2,  0,  0],
            [0, -2,  8, -2,  0],
            [1,  0, 10,  0,  1],
            [0, -2,  8, -2,  0],
            [0,  0, -2,  0,  0],
        ], dtype=torch.get_default_dtype()).view(4, 1, 5, 5) / 16.0

    def _calc_interpolated_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 1, "Input image must be grayscale"

        x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
        feats = F.conv2d(x_pad, self.kernels.to(x.device))
        feats = torch.cat((feats, x), dim=1)  # B, 5, H, W
        return feats


class DemosaicAHD(nn.Module):
    def __init__(self):
        """ Apply demosaicing to the input image using Adaptive Homogeneity-Directed method.

        References:
            - Hirakawa, Keigo, and Thomas W. Parks.
            "Adaptive homogeneity-directed demosaicing algorithm."
            IEEE transactions on image processing 14.3 (2005): 360-369.
            - https://github.com/12dmodel/camera_sim/blob/master/data_generation/ahd_demosaicking.py
        """
        super().__init__()

    def __call__(self, x: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        """ Apply demosaicing to the input image using Adaptive Homogeneity-Directed method.

        Args:
            x: Input bayer images.
            pattern_per_instance: pattern per each instance in input batch.

        Returns:
            torch.Tensor: Demosaiced images.
        """
        return super().__call__(x, pattern)

    def forward(self, x: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
