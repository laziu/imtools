from pathlib import Path

import torch
import numpy as np
import PIL.Image
import rawpy
import tifffile

from .convert import im2pil, im2numpy


def imload(*path) -> PIL.Image.Image:
    path = Path(*path).as_posix()
    return PIL.Image.open(path)


def imsave(image, *path):
    path = Path(*path).as_posix()
    im2pil(image).save(path)


def rawload(*path) -> rawpy.RawPy:
    path = Path(*path).as_posix()
    return rawpy.imread(path)


def tiffload(*path) -> np.ndarray:
    path = Path(*path).as_posix()
    return tifffile.imread(path)


def tiffsave(image, *path):
    path = Path(*path).as_posix()
    tifffile.imwrite(path, im2numpy(image))


def npload(*path, mode="r") -> np.ndarray:
    path = Path(*path).as_posix()
    return np.load(path, mmap_mode=mode)


def npsave(matrix, *path):
    path = Path(*path).as_posix()
    np.save(path, matrix)


def npzsave(matrices, *path, compress=True):
    path = Path(*path).as_posix()
    if compress:
        np.savez_compressed(path, **matrices)
    else:
        np.savez(path, **matrices)


def nptxtload(*path, dtype=float, delimiter=" ", comments="#") -> np.ndarray:
    path = Path(*path).as_posix()
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter, comments=comments)


def nptxtsave(matrix, *path,
              fmt="%.18e", delimiter=" ", newline="\n", header="", footer="", comments="#"):
    path = Path(*path).as_posix()
    np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter, newline=newline,
               header=header, footer=footer, comments=comments)


def ptload(*path, map_location=None) -> torch.Tensor:
    path = Path(*path).as_posix()
    return torch.load(path, map_location=map_location)


def ptsave(tensor, *path):
    path = Path(*path).as_posix()
    torch.save(tensor, path)
