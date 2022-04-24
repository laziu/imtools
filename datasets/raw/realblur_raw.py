from typing import Callable, Literal
from pathlib import Path
import csv
import json
from time import time

import torch
import torch.utils.data
import numpy as np

import utils
import utils.transforms
import utils.color
import utils.bayer


class RealBlurRaw(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, datalist_path: str, normalize: bool = True, transform: Callable = None):
        """ RealBlur RAW - RGB images.

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
        name = raw_path.split(".")[0]

        # lts = time()
        rgb = utils.im2tensor(utils.imload(self.data_dir / rgb_path))
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Load RGB: {dur:4f}s")

        raw = utils.im2tensor(utils.loadtiff(self.data_dir / raw_path))
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Load RAW: {dur:4f}s")

        tags = utils.loadexif(self.data_dir / raw_path)
        metadata = json.loads(str(tags["Image ImageDescription"]))
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Load EXIF: {dur:4f}s")

        bayer_pattern = metadata["bayer_pattern"]
        bayer_pattern = np.array([{"R": 0, "G": 1, "B": 2}[c] for c in list(bayer_pattern)]).reshape((2, 2))
        bayer_pattern = torch.as_tensor(bayer_pattern)
        H, W = raw.shape[-2:]
        bayer_mask = bayer_pattern.repeat(int(np.ceil(H / 2)), int(np.ceil(W / 2)))[:H, :W]

        white_balance = np.asarray(metadata["white_balance"])
        white_balance = torch.as_tensor(white_balance, dtype=torch.float32) / white_balance[1]

        color_matrix = np.asarray(metadata["color_matrix"]).astype(np.float32)
        color_matrix = torch.as_tensor(color_matrix, dtype=torch.float32)

        black_level = torch.as_tensor(metadata["black_level"])[:3]
        white_level = torch.as_tensor(metadata["white_level"])
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Load metadata: {dur:4f}s")

        camera_id = "SONY/ILCE-7RM3"

        if self.normalize:
            raw = utils.normalize(raw, black_level, white_level, bayer_pattern=bayer_pattern)
            rgb = utils.im2float(rgb)
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Normalize: {dur:4f}s")

        if self.transform:
            raw, rgb, bayer_mask = self.transform(raw, rgb, bayer_mask)
        # lts, dur = (cts := time()), cts - lts; print(f"RBR Transform: {dur:4f}s")

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
            "orientation": "",
        }
