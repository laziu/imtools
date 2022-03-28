from typing import TypeVar

import torch
import numpy as np
import PIL.Image


TArray = TypeVar("TArray", torch.Tensor, np.ndarray)
TImage = TypeVar("TImage", torch.Tensor, np.ndarray, PIL.Image.Image)


def ensure_tuple(x, len):
    try:
        return x[:len]
    except TypeError:
        return (x,) * len


class VoidModule:
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class AvgMeter:
    def __init__(self, format="{avg:.4f} (={sum:.4f}/{count:.0f})"):
        """ Computes and stores the average and current value. """
        self.format = format
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def add(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return self.format.format(**self.__dict__)
