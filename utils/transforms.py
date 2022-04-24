from typing import Literal, Union

import torchvision.transforms.functional as Ftrans
import numpy as np
from torchvision.transforms.functional import InterpolationMode

from .convert import numpy_to_tensor, tensor_to_numpy
from .misc import TArray, TImage


def resize(
    img: TImage,
    size: Union[int, tuple[int, int]],
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    max_size: int = None,
    antialias: bool = None,
) -> TImage:
    """ Resize the input image to the given size.

    Args:
        img: Image to be resized.
        size: Desired output size as (h, w) or a single int.
        If size is a single int, the smaller edge of the image will
        be matched to this number maintaining the aspect ratio.
        i.e. if h > w, then output size = (size * h/w, size).
        interpolation: Desired interpolation mode.
        max_size: if the longer edge of the image is larger than `max_size`
        after being resized according to `size`, then the image is resized
        again so that the longer edge is equal to `max_size`. This is only
        supported if `size` is an single int.
        antialias: antialias flag. If `img` is PIL Image than antialias is
        always used.
    """
    interpolation = (InterpolationMode.NEAREST if interpolation == "nearest" else
                     InterpolationMode.BILINEAR if interpolation == "bilinear" else
                     InterpolationMode.BICUBIC if interpolation == "bicubic" else None)

    if not interpolation:
        raise ValueError(f"Unknown interpolation mode: {interpolation}")

    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.resize(
        img, size,
        interpolation=interpolation,
        max_size=max_size, antialias=antialias,
    )

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class Resize:
    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
        max_size: int = None,
        antialias: bool = None,
    ) -> None:
        """ Resize the input image to the given size, see `resize` for more details. """
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, image: TImage) -> TImage:
        return resize(image, size=self.size, interpolation=self.interpolation,
                      max_size=self.max_size, antialias=self.antialias)


def pad(
    img: TArray,
    padding: Union[int, tuple[int, int], tuple[int, int, int, int]],
    fill: int = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
) -> TArray:
    """ Pad the image.

    Args:
        img: Image to be padded.
        padding: Padding on each border, as (all_side), (left/right, top/bottom),
        or (left, top, right, bottom) shape.
        fill: Pixel fill value for constant fill.
        padding_mode: Type of padding.
    """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.pad(img, padding, fill=fill, padding_mode=padding_mode)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class Pad:
    def __init__(
        self,
        padding: Union[int, tuple[int, int], tuple[int, int, int, int]],
        fill: int = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ) -> None:
        """ Pad the image, see `pad` for more details. """
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image: TImage) -> TImage:
        return pad(image, padding=self.padding, fill=self.fill, padding_mode=self.padding_mode)


def crop(img: TArray, top: int, left: int, height: int, width: int) -> TArray:
    """ Crop the image.

    Args:
        img: Image to be cropped.
        top: Top of the top left corner of the crop box.
        left: Left of the top left corner of the crop box.
        height: Height of the crop box.
        width: Width of the crop box.
    """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.crop(img, top, left, height, width)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class Crop:
    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        """ Crop the image, see `crop` for more details. """
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image: TImage) -> TImage:
        return crop(image, top=self.top, left=self.left, height=self.height, width=self.width)


def center_crop(img: TArray, output_size: Union[int, tuple[int, int]]) -> TArray:
    """ Crop the image at the center.

    Args:
        img: Image to be cropped.
        output_size: Expected output size as (h, w) or (all_side).
    """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.center_crop(img, output_size)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class CenterCrop:
    def __init__(self, output_size: Union[int, tuple[int, int]]) -> None:
        """ Crop the image at the center, see `center_crop` for more details. """
        self.output_size = output_size

    def __call__(self, image: TImage) -> TImage:
        return center_crop(image, output_size=self.output_size)


def hflip(img: TArray) -> TArray:
    """ Horizontally flip the image. """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.hflip(img)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class HorizontalFlip:
    """ Horizontally flip the image, see `hflip` for more details. """

    def __call__(self, image: TImage) -> TImage:
        return hflip(image)


def vflip(img: TArray) -> TArray:
    """ Vertically flip the image. """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.vflip(img)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class VerticalFlip:
    """ Vertically flip the image, see `vflip` for more details. """

    def __call__(self, image: TImage) -> TImage:
        return vflip(image)


def rotate(
    img: TArray,
    angle: float,
    interpolation: Literal["nearest", "bilinear"] = "bilinear",
    expand: bool = False,
    center: tuple[int, int] = None,
    fill: Union[float, list[float]] = None,
) -> TArray:
    """ Rotate the image.

    Args:
        img: Image to be rotated.
        angle: Rotation angle in degrees.
        interpolation: Interpolation mode.
        expand: If True, expands the output image to fit the rotated image.
        center: The center of rotation from the top-left corner.
        fill: Pixel fill value for constant fill.
    """
    interpolation = (InterpolationMode.NEAREST if interpolation == "nearest" else
                     InterpolationMode.BILINEAR if interpolation == "bilinear" else None)
    if not interpolation:
        raise ValueError(f"Unknown interpolation mode: {interpolation}")

    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.rotate(img, angle, interpolation, expand, center, fill)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class Rotate:
    def __init__(
        self,
        angle: float,
        interpolation: Literal["nearest", "bilinear"] = "bilinear",
        expand: bool = False,
        center: tuple[int, int] = None,
        fill: Union[float, list[float]] = None,
    ) -> None:
        """ Rotate the image, see `rotate` for more details. """
        super().__init__()
        self.angle = angle
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, image: TImage) -> TImage:
        return rotate(image, angle=self.angle, interpolation=self.interpolation,
                      expand=self.expand, center=self.center, fill=self.fill)


def erase(img: TArray, i: int, j: int, h: int, w: int, v: TArray, inplace: bool = False) -> TArray:
    """ Erase the given area of the input image with given value.

    Args:
        img: Image to be erased.
        i: i in (i, j) of the upper left corner.
        j: j in (i, j) of the upper left corner.
        h: Height of the area to be erased.
        w: Width of the area to be erased.
        v: Fill value.
        inplace: If True, erase the area in-place.
    """
    if is_numpy := isinstance(img, np.ndarray):
        dtype = img.dtype
        img = numpy_to_tensor(img)

    img = Ftrans.erase(img, i, j, h, w, v, inplace)

    if is_numpy:
        img = tensor_to_numpy(img).astype(dtype)

    return img


class Erase:
    def __init__(self, i: int, j: int, h: int, w: int, v: TArray, inplace: bool = False) -> None:
        """ Erase the given area of the input image with given value, see `erase` for more details. """
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.v = v
        self.inplace = inplace

    def __call__(self, image: TImage) -> TImage:
        return erase(image, i=self.i, j=self.j, h=self.h, w=self.w, v=self.v, inplace=self.inplace)
