import typing
import numbers
from pathlib import Path
import multiprocessing as mp

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import numpy as np
from tqdm.auto import tqdm


TCropArea: typing.TypeAlias = tuple[int, int, int, int]


class CropConfig:
    @classmethod
    def setup_size(cls, size) -> tuple[int, int]:
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, typing.Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError("size should be (h, w) tuple or a single number")

        return size

    def __call__(self, crop_area: TCropArea) -> TCropArea:
        return crop_area


class RandomCropConfig(CropConfig):
    def __init__(self, size: int | tuple[int, int]):
        """ Random crop configuration.

        Args:
            size: crop size.
            unify_bayer: if True, unify bayer pattern to RGGB.
        """
        self.size = self.setup_size(size)

    def __call__(self, crop_area: TCropArea) -> TCropArea:
        ci, cj, ch, cw = crop_area
        th, tw = self.size

        assert ch >= th and cw >= tw, "Crop area is smaller than crop size"

        ti = ci + torch.randint(0, ch - th, (1, )).item()
        tj = cj + torch.randint(0, cw - tw, (1, )).item()

        return ti, tj, th, tw


class CenterCropConfig(CropConfig):
    def __init__(self, size: int | tuple[int, int]):
        """ Center crop configuration.

        Args:
            size: crop size.
            unify_bayer: if True, unify bayer pattern to RGGB.
        """
        self.size = self.setup_size(size)

    def __call__(self, crop_area: TCropArea) -> TCropArea:
        ci, cj, ch, cw = crop_area
        th, tw = self.size

        assert ch >= th and cw >= tw, "Crop area is smaller than crop size"

        ti = ci + ((ch - th) // 2)
        tj = cj + ((cw - tw) // 2)

        return ti, tj, th, tw


class ComposeCropConfig(CropConfig):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, crop_area: TCropArea) -> TCropArea:
        for config in self.configs:
            crop_area = config(crop_area)
        return crop_area


class CachedCropDataset(torch.utils.data.Dataset):
    MULTIPROCESSING = 4
    PATCH_SIZE = 512  # should be even number
    LAYER_KEYS = ["raw", "rgb", "bayer_mask"]

    def __init__(self, dataset: torch.utils.data.Dataset, cache_dir: str,
                 crop_config: CropConfig = None, unify_bayer: bool = False,
                 skip_check: bool = False, **kwargs):
        """ Cached dataset.

        Args:
            dataset: dataset to be cached.
            cache_dir: cache directory.
            crop_config: crop configuration.
            unify_bayer: if True, unify bayer pattern to RGGB.
                NOTE: set crop size to odd number since this option cut 1px from each side.
            skip_check: Set True if cache is already exist.
        """
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.crop_config = crop_config or CropConfig()
        self.unify_bayer = unify_bayer

        if "MULTIPROCESSING" in kwargs:
            self.MULTIPROCESSING = kwargs["MULTIPROCESSING"]
        if "PATCH_SIZE" in kwargs:
            self.PATCH_SIZE = kwargs["PATCH_SIZE"]
        if "LAYER_KEYS" in kwargs:
            self.LAYER_KEYS = kwargs["LAYER_KEYS"]

        if not skip_check:
            self.cache_all()

    def __len__(self):
        return len(self.dataset)

    def get_inst_name(self, idx: int) -> Path:
        raise NotImplementedError

    def get_patch_name(self, i: int, j: int) -> str:
        return f"patch_{i:06d}_{j:06d}_{self.PATCH_SIZE}"

    def cache_item(self, idx: int):
        cache_path = self.cache_dir / self.get_inst_name(idx)
        if cache_path.exists():
            return False
        cache_path.mkdir(parents=True, exist_ok=True)

        item = self.dataset[idx]
        layers:   dict[str, torch.Tensor] = {k: v for k, v in item.items() if k in self.LAYER_KEYS}
        metadata: dict[str, torch.Tensor] = {k: v for k, v in item.items() if k not in self.LAYER_KEYS}

        ih, iw = list(layers.values())[0].shape[-2:]
        assert all([v.shape[-2:] == (ih, iw) for v in layers.values()]), \
            f"Image size mismatch: {[v.shape[-2:] for v in layers.values()]}"
        metadata["size"] = (ih, iw)

        for i in range(0, ih, self.PATCH_SIZE):
            ni = i + self.PATCH_SIZE
            for j in range(0, iw, self.PATCH_SIZE):
                nj = j + self.PATCH_SIZE
                patch = {k: v[..., i:ni, j:nj].clone().numpy() for k, v in layers.items()}
                np.savez_compressed((cache_path / self.get_patch_name(i, j)).as_posix() + ".npz", **patch)

        torch.save(metadata, (cache_path / "metadata.pt").as_posix())
        return True

    def cache_all(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.MULTIPROCESSING:
            with mp.Pool(self.MULTIPROCESSING) as pool:
                error_last = None

                def error_callback(e):
                    nonlocal error_last
                    print(e)
                    error_last = e

                promises = [pool.apply_async(self.cache_item, args=(idx,), error_callback=error_callback)
                            for idx in range(len(self))]
                for p in tqdm(promises, desc=f"Caching dataset {self.cache_dir.name}"):
                    p.get()
                if error_last:
                    raise error_last
        else:
            for idx in tqdm(range(len(self)), desc=f"Caching to {self.cache_dir.name}"):
                self.cache_item(idx)

    def load_item(self, idx: int):
        cache_path = self.cache_dir / self.get_inst_name(idx)

        metadata = torch.load((cache_path / "metadata.pt").as_posix())

        ih, iw = metadata["size"]
        ci, cj, ch, cw = self.crop_config((0, 0, ih, iw))

        pi = (ci // self.PATCH_SIZE) * self.PATCH_SIZE
        pj = (cj // self.PATCH_SIZE) * self.PATCH_SIZE

        patch = []
        for i in range(pi, ci + ch, self.PATCH_SIZE):
            patch_i = []
            for j in range(pj, cj + cw, self.PATCH_SIZE):
                patch_ij = np.load((cache_path / self.get_patch_name(i, j)).as_posix() + ".npz")
                patch_ij = {k: torch.from_numpy(patch_ij[k]) for k in self.LAYER_KEYS}

                patch_i.append(patch_ij)

            patch_i = {k: torch.cat([patch_ij[k] for patch_ij in patch_i], dim=-1)
                       for k in patch_i[0].keys()}

            patch.append(patch_i)

        patch = {k: torch.cat([patch_i[k] for patch_i in patch], dim=-2)
                 for k in patch[0].keys()}

        patch = {k: TF.crop(layer, ci - pi, cj - pj, ch, cw)
                 for k, layer in patch.items()}

        return patch | metadata
