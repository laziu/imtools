import torch
import torch.utils.data
import torchvision.transforms
import torchvision.transforms.functional as TF

from .realblur_raw import RealBlurRaw
from .raise_raw import RAISERaw


class PairCompose(torchvision.transforms.Compose):
    def __call__(self, raw: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor):
        for transform in self.transforms:
            raw, rgb, mask = transform(raw, rgb, mask)
        return raw, rgb, mask


class PairRandomCrop(torchvision.transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", unify_bayer=False):
        """ Randomly crop a pair of images.

        Args:
            size: crop size.
            padding: padding size.
            pad_if_needed: pad if needed.
            fill: fill value for padding.
            padding_mode: padding mode.
            unify_bayer: if True, unify bayer pattern to RGGB.
        """
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.unify_bayer = unify_bayer

    def __call__(self, raw: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor):
        # print(raw.shape, rgb.shape, mask.shape)
        if self.padding is not None:
            raw  = TF.pad(raw,  self.padding, self.fill, self.padding_mode)
            rgb  = TF.pad(rgb,  self.padding, self.fill, self.padding_mode)
            mask = TF.pad(mask, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and raw.size[0] < self.size[1]:
            raw  = TF.pad(raw,  (self.size[1] - raw.size[0],  0), self.fill, self.padding_mode)
            rgb  = TF.pad(rgb,  (self.size[1] - rgb.size[0],  0), self.fill, self.padding_mode)
            mask = TF.pad(mask, (self.size[1] - mask.size[0], 0), self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and raw.size[1] < self.size[0]:
            raw  = TF.pad(raw,  (0, self.size[0] - raw.size[1]),  self.fill, self.padding_mode)
            rgb  = TF.pad(rgb,  (0, self.size[0] - rgb.size[1]),  self.fill, self.padding_mode)
            mask = TF.pad(mask, (0, self.size[0] - mask.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(rgb, self.size)
        if self.unify_bayer:
            # print(raw.shape, rgb.shape, mask.shape, i, j, h, w)
            if mask[..., i, j] == 1:
                i = i + 1 if i == 0 else i - 1
            if mask[..., i, j] == 2:
                i = i + 1 if i == 0 else i - 1
                j = j + 1 if j == 0 else j - 1

        raw  = TF.crop(raw,  i, j, h, w)
        rgb  = TF.crop(rgb,  i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return raw, rgb, mask


class PairCenterCrop(torchvision.transforms.CenterCrop):
    def __init__(self, size, unify_bayer: bool = False):
        """ Center crop a pair of images.

        Args:
            size: crop size.
            unify_bayer: if True, unify bayer pattern to RGGB.
        """
        super().__init__(size)
        self.unify_bayer = unify_bayer

    def __call__(self, raw: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor):
        ih, iw = rgb.shape[-2:]
        ch, cw = self.size

        ci = int(round((ih - ch) / 2.))
        cj = int(round((iw - cw) / 2.))

        if self.unify_bayer:
            if mask[..., ci, cj] == 1:
                ci = ci + 1 if ci == 0 else ci - 1
            if mask[..., ci, cj] == 2:
                ci = ci + 1 if ci == 0 else ci - 1
                cj = cj + 1 if cj == 0 else cj - 1

        raw  = TF.crop(raw,  ci, cj, ch, cw)
        rgb  = TF.crop(rgb,  ci, cj, ch, cw)
        mask = TF.crop(mask, ci, cj, ch, cw)

        return raw, rgb, mask


class Concatenate(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        """ Concatenate multiple datasets.

        Args:
            *datasets: datasets to concatenate.
        """
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, index: int):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")
