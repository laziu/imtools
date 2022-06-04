from pathlib import Path
from time import sleep
import json

import torch
import numpy as np
import PIL.Image
import PIL.ImageFile
import rawpy
import tifffile
import exifread
import scipy.io

from .convert import im2pil, im2numpy
from ._typing.rawpy import RawPy

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def imload(*path) -> PIL.Image.Image:
    path = Path(*path).as_posix()
    for i in range(3):
        try:
            with PIL.Image.open(path) as im:
                return np.asarray(im)
        except OSError:
            sleep(i + 1)
    with PIL.Image.open(path) as im:
        return np.asarray(im)


def imsave(image, *path, **kwargs):
    path = Path(*path).as_posix()
    with im2pil(image) as f:
        f.save(path, **kwargs)


def loadraw(*path) -> RawPy:
    path = Path(*path).as_posix()
    try:
        return rawpy.imread(path)
    except Exception as e:
        raise OSError(f"Failed to load raw image {path}") from e


def loadexif(*path) -> dict[str, str]:
    path = Path(*path).as_posix()
    with open(path, "rb") as f:
        tags = exifread.process_file(f)
    return tags


def loadtiff(*path) -> np.ndarray:
    path = Path(*path).as_posix()
    try:
        return tifffile.imread(path)
    except Exception as e:
        raise OSError(f"Failed to load tiff image {path}") from e


def savetiff(image, *path, **kwargs):
    path = Path(*path).as_posix()
    tifffile.imwrite(path, im2numpy(image), **kwargs)


def loadnpy(*path, mode="r") -> np.ndarray:
    path = Path(*path).as_posix()
    return np.load(path, mmap_mode=mode)


def savenpy(matrix, *path):
    path = Path(*path).as_posix()
    np.save(path, matrix)


def savenpz(matrices, *path, compress=True):
    path = Path(*path).as_posix()
    if compress:
        np.savez_compressed(path, **matrices)
    else:
        np.savez(path, **matrices)


def loadnptxt(*path, dtype=float, delimiter=" ", comments="#") -> np.ndarray:
    path = Path(*path).as_posix()
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter, comments=comments)


def savenptxt(matrix, *path,
              fmt="%.18e", delimiter=" ", newline="\n", header="", footer="", comments="#"):
    path = Path(*path).as_posix()
    np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter, newline=newline,
               header=header, footer=footer, comments=comments)


def loadmat(*path) -> dict[str, np.ndarray]:
    path = Path(*path).as_posix()
    return scipy.io.loadmat(path)


def savemat(matrices, *path):
    path = Path(*path).as_posix()
    scipy.io.savemat(path, matrices)


def loadpt(*path, map_location=None) -> torch.Tensor:
    path = Path(*path).as_posix()
    return torch.load(path, map_location=map_location)


def savept(tensor, *path):
    path = Path(*path).as_posix()
    torch.save(tensor, path)


def loadjson(*path):
    path = Path(*path).as_posix()
    with open(path, "r") as f:
        return json.load(f)


def savejson(obj, *path):
    path = Path(*path).as_posix()
    with open(path, "w") as f:
        json.dump(obj, f)
