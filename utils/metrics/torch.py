import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import kornia.losses
from torch import Tensor

from ..convert import reshape_tensor


def ssim_loss(x: Tensor, y: Tensor, window_size: int = 11) -> Tensor:
    """ Compute the Structrual dissimilarity (DSSIM), i.e. SSIM loss.

    Args:
        x: Tensor of shape (B, C, H, W)
        y: Tensor of shape (B, C, H, W)
        window_size: Size of the window to use.
    """
    x = reshape_tensor(x)
    y = reshape_tensor(y)
    return kornia.losses.ssim_loss(x, y, window_size=window_size)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        """ Module version of `ssim_loss`. """
        super().__init__()
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=window_size)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """ Module version of `ssim_loss`. """
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.ssim_loss(x, y)


def ssim(x: Tensor, y: Tensor, window_size: int = 11) -> Tensor:
    """ Compute the Structural similarity (SSIM).

    Args:
        x: Tensor of shape (B, C, H, W)
        y: Tensor of shape (B, C, H, W)
        window_size: Size of the window to use.
    """
    return 1 - 2 * ssim_loss(x, y, window_size=window_size)


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11):
        """ Module version of `ssim`. """
        super().__init__()
        self.dssim = SSIMLoss(window_size=window_size)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """ Module version of `ssim`. """
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - 2 * self.dssim(x, y)


def psnr_loss(x: Tensor, y: Tensor, max_val: float = 1.) -> Tensor:
    """ Compute the negative Peak Signal-to-Noise Ratio (-PSNR), i.e. PSNR loss. """
    x = reshape_tensor(x)
    y = reshape_tensor(y)
    return kornia.losses.psnr_loss(x, y, max_val=max_val)


class PSNRLoss(nn.Module):
    def __init__(self, max_val: float = 1.):
        """ Module version of `psnr_loss`. """
        super().__init__()
        self.psnr_loss = kornia.losses.PSNRLoss(max_val=max_val)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """ Module version of `psnr_loss`. """
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.psnr_loss(x, y)


def psnr(x: Tensor, y: Tensor, max_val: float = 1.) -> Tensor:
    """ Compute the Peak Signal-to-Noise Ratio (PSNR).

    Args:
        x: input image
        y: label image
        max_val: Maximum value of the input image
    """
    return -psnr_loss(x, y, max_val=max_val)


class PSNR(nn.Module):
    def __init__(self, max_val: float = 1.):
        """ Module version of `psnr`. """
        super().__init__()
        self.neg_psnr = PSNRLoss(max_val=max_val)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """ Module version of `psnr`. """
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return -self.neg_psnr(x, y)


class VGGLoss(nn.Module):
    def __init__(self,
                 feature_layers: list[int] = [0, 1, 2, 3],
                 style_layers: list[int] = [],
                 resize=True):
        """Compute a loss based on VGG Perceptual loss.

        Args:
            feature_layers: VGG16 layers to use in feature perceptual loss.
            style_layers: VGG16 layers to use in style perceptual loss.
            resize: resize into fixed size before calculate loss if True.
            device: target device.
        """
        super().__init__()
        self.feature_layers = feature_layers
        self.style_layers = style_layers
        self.resize = resize

        blocks = [
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),
            torchvision.models.vgg16(pretrained=True).features[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.mean: Tensor
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std: Tensor
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """ Compute the VGG perceptual loss. """
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        if self.resize:
            x = F.interpolate(x, mode="bilinear", size=(224, 224), align_corners=False)
            y = F.interpolate(y, mode="bilinear", size=(224, 224), align_corners=False)

        loss = 0.0
        for i, block in enumerate(self.blocks):
            x: Tensor = block(x)
            y: Tensor = block(y)
            if i in self.feature_layers:
                loss += F.l1_loss(x, y)
            if i in self.style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += F.l1_loss(gram_x, gram_y)

        return loss


def vgg_loss(
    x: Tensor,
    y: Tensor,
    feature_layers: list[int] = [0, 1, 2, 3],
    style_layers: list[int] = [],
    resize: bool = True,
) -> Tensor:
    """ Functional version of `VGGLoss`. """
    x = reshape_tensor(x)
    y = reshape_tensor(y)
    module = VGGLoss(feature_layers=feature_layers,
                     style_layers=style_layers,
                     resize=resize).to(x.device)
    return module(x, y)
