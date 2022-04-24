from typing import Callable, Literal
from pathlib import Path
import csv
from time import time
from math import floor, ceil

import torch
import torch.utils.data
import numpy as np

import utils
import utils.transforms
import utils.color
import utils.bayer


# such airhead code
SRGB_D50_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
SRGB_D65_TO_XYZ = np.array([[0.4360747, 0.3850649, 0.1430804],
                            [0.2225045, 0.7168786, 0.0606169],
                            [0.0139322, 0.0971045, 0.7141733]])
D50_TO_D65 = SRGB_D50_TO_XYZ @ np.linalg.inv(SRGB_D65_TO_XYZ)
D65_TO_D50 = np.linalg.inv(D50_TO_D65)


class RAISERaw(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, datalist_path: str, normalize: bool = True, transform: Callable = None):
        """ RAISE RAW - RGB images.

        Args:
            data_dir: path to the directory containing the dataset.
            datalist_path: path to the csv file containing the list of pairs.
            normalize: normalize raw and rgb images to [0, 1].
            transform: transform to apply to the data, must be able to handle (raw, rgb, bayer_mask) as input.

        Returns: dict
            - raw: raw image tensor.
            - rgb: rgb image tensor.
            - bayer_pattern: bayer pattern as str.
            - white_balance: white balance.
            - color_matrix: color matrix.
            - black_level: black level.
            - white_level: white level.
            - camera_id: camera id.
            - orientation: orientation of the image.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.transform = transform

        with open(datalist_path, "r") as f:
            self.datalist = list(csv.reader(f))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx: int):
        raw_path, rgb_path = self.datalist[idx]
        name = Path(raw_path).stem

        rgb = utils.im2tensor(utils.imload(self.data_dir / rgb_path))

        raw_file = utils.loadraw(self.data_dir / raw_path)
        raw = utils.im2tensor(raw_file.raw_image_visible)

        tags = utils.loadexif(self.data_dir / raw_path)
        camera_id = f"{tags['Image Make']}/{tags['Image Model']}"
        orientation = str(tags["Image Orientation"])

        bayer_mask = raw_file.raw_pattern
        bayer_mask = torch.as_tensor(bayer_mask)
        bayer_mask = bayer_mask - 2 * (bayer_mask == 3).to(int)
        H, W = raw.shape[-2:]
        bayer_mask = bayer_mask.repeat(ceil(H / 2), ceil(W / 2))[:H, :W]

        if camera_id == "NIKON CORPORATION/NIKON D7000":
            raw = raw[..., 8:-8, 10:-10]
            bayer_mask = bayer_mask[..., 8:-8, 10:-10]
        elif camera_id == "NIKON CORPORATION/NIKON D90":
            raw = raw[..., 10:-10, 11:-11]
            bayer_mask = bayer_mask[..., 10:-10, 11:-11]
        elif camera_id == "NIKON CORPORATION/NIKON D40":
            raw = raw[..., 7:-7, 15:-16]
            bayer_mask = bayer_mask[..., 7:-7, 15:-16]

        # raw = self._rotate_as_tag(raw, orientation)
        # bayer_mask = self._rotate_as_tag(bayer_mask, orientation)
        rgb = self._reverse_rotate_as_tag(rgb, orientation)
        assert raw.shape[-2:] == rgb.shape[-2:], f"{raw.shape} != {rgb.shape}; {camera_id}/{orientation}"

        white_balance = np.array(raw_file.camera_whitebalance[:3])
        white_balance = torch.as_tensor(white_balance, dtype=torch.float32) / white_balance[1]

        color_matrix = np.array(raw_file.rgb_xyz_matrix)[:3]
        color_matrix /= np.sum(color_matrix, axis=1, keepdims=True)
        color_matrix = D50_TO_D65 @ np.linalg.inv(color_matrix)
        color_matrix = torch.as_tensor(color_matrix, dtype=torch.float32)

        black_level = torch.as_tensor(raw_file.black_level_per_channel)[:3]
        white_level = torch.as_tensor(raw_file.white_level)

        raw_file.close()

        if self.normalize:
            raw = utils.normalize(raw, black_level, white_level, bayer_pattern=bayer_mask)
            rgb = utils.im2float(rgb)

        if self.transform:
            raw, rgb, bayer_mask = self.transform(raw, rgb, bayer_mask)

        return {
            "name": name,
            "raw": raw,
            "rgb": rgb,
            "bayer_mask": bayer_mask,
            "white_balance": white_balance,
            "color_matrix": color_matrix,
            "black_level": black_level,
            "white_level": white_level,
            "camera_id": camera_id,
            "orientation": orientation,
        }

    @classmethod
    def _rotate_as_tag(cls, img: torch.Tensor, orientation: str):
        """ Rotate image according to the orientation tag. """
        if orientation == "Mirrored horizontal":
            img = torch.flip(img, axis=-1)
        elif orientation == "Rotated 180":
            img = torch.rot90(img, k=2, dims=(-2, -1))
        elif orientation == "Mirrored vertical":
            img = torch.flip(img, axis=-2)
        elif orientation == "Mirrored horizontal then rotated 90 CCW":
            img = torch.rot90(torch.flip(img, axis=-1), k=1, dims=(-2, -1))
        elif orientation == "Rotated 90 CW":
            img = torch.rot90(img, k=3, dims=(-2, -1))
        elif orientation == "Mirrored horizontal then rotated 90 CW":
            img = torch.rot90(torch.flip(img, axis=-1), k=3, dims=(-2, -1))
        elif orientation == "Rotated 90 CCW":
            img = torch.rot90(img, k=1, dims=(-2, -1))
        elif orientation != "Horizontal (normal)":
            raise ValueError(f"Unknown orientation: {orientation}")
        return img

    @classmethod
    def _reverse_rotate_as_tag(cls, img: torch.Tensor, orientation: str):
        """ Reverse rotate image according to the orientation tag. """
        if orientation == "Mirrored horizontal":
            img = torch.flip(img, axis=-1)
        elif orientation == "Rotated 180":
            img = torch.rot90(img, k=2, dims=(-2, -1))
        elif orientation == "Mirrored vertical":
            img = torch.flip(img, axis=-2)
        elif orientation == "Mirrored horizontal then rotated 90 CCW":
            img = torch.rot90(img, k=3, dims=(-2, -1))
            img = torch.flip(img, axis=-1)
        elif orientation == "Rotated 90 CW":
            img = torch.rot90(img, k=1, dims=(-2, -1))
        elif orientation == "Mirrored horizontal then rotated 90 CW":
            img = torch.rot90(img, k=1, dims=(-2, -1))
            img = torch.flip(img, axis=-1)
        elif orientation == "Rotated 90 CCW":
            img = torch.rot90(img, k=3, dims=(-2, -1))
        elif orientation != "Horizontal (normal)":
            raise ValueError(f"Unknown orientation: {orientation}")
        return img
