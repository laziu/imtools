import typing
from pathlib import Path

import torch
import torch.utils.data

import utils

from ..raise_raw import RAISERaw as _RAISERaw_Original
from .common import CropConfig, CachedCropDataset


class RAISERaw(CachedCropDataset):
    def __init__(self, data_dir: str, datalist_path: str, cache_dir: str,
                 crop_config: CropConfig = None, unify_bayer: bool = False,
                 normalize: bool = True, post_transform: typing.Callable = None,
                 skip_check: bool = False, **kwargs):
        """ RAISE RAW - RGB dataset, with cache.

        Args:
            data_dir: path to the directory containing the dataset.
            datalist_path: path to the csv file containing the list of pairs.
            cache_dir: path to the directory to save cache.
            crop_config: crop configuration.
            unify_bayer: if True, unify bayer pattern to RGGB.
                NOTE: set crop size to odd number since this option cut 1px from each side.
            normalize: normalize raw and rgb images to [0, 1].
            post_transform: transform to apply to the data, must be able to handle (raw, rgb, bayer_mask) as input.
            skip_check: Set True if cache is already exist.

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
        super().__init__(
            dataset=_RAISERaw_Original(data_dir, datalist_path, normalize=False),
            cache_dir=cache_dir,
            crop_config=crop_config,
            unify_bayer=unify_bayer,
            skip_check=skip_check,
            **kwargs,
        )
        self.normalize      = normalize
        self.post_transform = post_transform

    def get_inst_name(self, idx: int) -> Path:
        raw_path = self.dataset.datalist[idx][0]
        return Path(raw_path).stem

    def __getitem__(self, idx: int):
        item = self.load_item(idx)

        if self.unify_bayer:
            if item["bayer_mask"][..., 0, 0] == 0:
                for k in self.LAYER_KEYS:
                    item[k] = item[k][..., :-1, :-1]
            elif item["bayer_mask"][..., 0, 1] == 0:
                for k in self.LAYER_KEYS:
                    item[k] = item[k][..., :-1, 1:]
            elif item["bayer_mask"][..., 1, 0] == 0:
                for k in self.LAYER_KEYS:
                    item[k] = item[k][..., 1:, :-1]
            elif item["bayer_mask"][..., 1, 1] == 0:
                for k in self.LAYER_KEYS:
                    item[k] = item[k][..., 1:, 1:]
            else:
                raise ValueError("Invalid bayer mask")

        if self.normalize:
            item["raw"] = utils.normalize(item["raw"], item["black_level"], item["white_level"],
                                          bayer_pattern=item["bayer_mask"])
            item["rgb"] = utils.im2float(item["rgb"])

        if self.post_transform:
            item["raw"], item["rgb"], item["bayer_mask"] = \
                self.post_transform(item["raw"], item["rgb"], item["bayer_mask"])

        return item
