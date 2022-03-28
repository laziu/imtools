from typing import Union
from warnings import warn

import torch
import torchvision.transforms.functional as Ftrans
import numpy as np
import PIL.Image

from .misc import TArray


def as_torch_dtype(dtype: Union[torch.dtype, np.dtype, type, str]) -> torch.dtype:
    """ Convert the dtype to torch.dtype. """
    if dtype in ["bool", bool, np.bool8, torch.bool]:
        return torch.bool
    if dtype in ["uint8", "u8", np.uint8, torch.uint8]:
        return torch.uint8
    if dtype in ["int8", "i8", np.int8, torch.int8]:
        return torch.int8
    if dtype in ["int16", "i16", np.int16, torch.int16]:
        return torch.int16
    if dtype in ["int32", "i32", np.int32, torch.int32]:
        return torch.int32
    if dtype in ["int64", "i64", int, np.int64, torch.int64]:
        return torch.int64
    if dtype in ["float16", "f16", np.float16, torch.float16]:
        return torch.float16
    if dtype in ["float32", "f32", np.float32, torch.float32]:
        return torch.float32
    if dtype in ["float64", "f64", float, np.float64, torch.float64]:
        return torch.float64
    if dtype in ["complex64", "c64", np.complex64, torch.complex64]:
        return torch.complex64
    if dtype in ["complex128", "c128", np.complex128, torch.complex128]:
        return torch.complex128
    raise ValueError(f"Unknown dtype: {dtype}")


def as_np_dtype(dtype: Union[torch.dtype, np.dtype, type, str]) -> np.dtype:
    """ Convert the dtype to numpy dtype. """
    if dtype in ["bool", bool, np.bool8, torch.bool]:
        return np.bool8
    if dtype in ["uint8", "u8", np.uint8, torch.uint8]:
        return np.uint8
    if dtype in ["uint16", "u16", np.uint16]:
        return np.uint16
    if dtype in ["uint32", "u32", np.uint32]:
        return np.uint32
    if dtype in ["uint64", "u64", np.uint64]:
        return np.uint64
    if dtype in ["uint128", "u128", np.uint128]:
        return np.uint128
    if dtype in ["uint256", "u256", np.uint256]:
        return np.uint256
    if dtype in ["int8", "i8", np.int8, torch.int8]:
        return np.int8
    if dtype in ["int16", "i16", np.int16, torch.int16]:
        return np.int16
    if dtype in ["int32", "i32", np.int32, torch.int32]:
        return np.int32
    if dtype in ["int64", "i64", int, np.int64, torch.int64]:
        return np.int64
    if dtype in ["int128", "i128", np.int128]:
        return np.int128
    if dtype in ["int256", "i256", np.int256]:
        return np.int256
    if dtype in ["float16", "f16", np.float16, torch.float16]:
        return np.float16
    if dtype in ["float32", "f32", np.float32, torch.float32]:
        return np.float32
    if dtype in ["float64", "f64", float, np.float64, torch.float64]:
        return np.float64
    if dtype in ["float80", "f80", np.float80]:
        return np.float80
    if dtype in ["float96", "f96", np.float96]:
        return np.float96
    if dtype in ["float128", "f128", np.float128]:
        return np.float128
    if dtype in ["float256", "f256", np.float256]:
        return np.float256
    if dtype in ["complex64", "c64", np.complex64, torch.complex64]:
        return np.complex64
    if dtype in ["complex128", "c128", np.complex128, torch.complex128]:
        return np.complex128
    if dtype in ["complex160", "c160", np.complex160]:
        return np.complex160
    if dtype in ["complex192", "c192", np.complex192]:
        return np.complex192
    if dtype in ["complex256", "c256", np.complex256]:
        return np.complex256
    if dtype in ["complex512", "c512", np.complex512]:
        return np.complex512
    raise ValueError(f"Unknown dtype: {dtype}")


def is_floating_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in [torch.float16, torch.float32, torch.float64]


def is_integer_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def is_complex_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in [torch.complex64, torch.complex128]


def is_floating_dtype(dtype: Union[torch.dtype, np.dtype, type, str]) -> bool:
    return np.issubdtype(as_np_dtype(dtype), np.floating)


def is_integer_dtype(dtype: Union[torch.dtype, np.dtype, type, str]) -> bool:
    return np.issubdtype(as_np_dtype(dtype), np.integer)


def is_complex_dtype(dtype: Union[torch.dtype, np.dtype, type, str]) -> bool:
    return np.issubdtype(as_np_dtype(dtype), np.complexfloating)


def is_incompatible_dtype(dtype: np.dtype) -> bool:
    """ Return True if the numpy dtype is incompatible with torch. """
    return dtype in [np.uint16, np.uint32, np.uint64,
                     np.uint128, np.uint256, np.int128, np.int256,
                     np.float80, np.float96, np.float128, np.float256,
                     np.complex160, np.complex192, np.complex256, np.complex512]


def as_signed(x: TArray) -> TArray:
    """ Convert the unsigned integer array to be signed. """
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.uint8:
            return x.to(torch.int8 if x.max() < 2**7 else torch.int16)
    else:
        if x.dtype == np.uint8:
            return x.astype(np.int8 if x.max() < 2**7 else np.int16)
        elif x.dtype == np.uint16:
            return x.astype(np.int16 if x.max() < 2**15 else np.int32)
        elif x.dtype == np.uint32:
            return x.astype(np.int32 if x.max() < 2**31 else np.int64)
        elif x.dtype == np.uint64:
            return x.astype(np.int64 if x.max() < 2**63 else np.int128)
        elif x.dtype == np.uint128:
            return x.astype(np.int128 if x.max() < 2**127 else np.int256)
        elif x.dtype == np.uint256:
            if x.max() >= 2**255:
                raise ValueError("Input array is too large to be converted to signed.")
            return x.astype(np.int256)

    raise ValueError(f"Unsupported dtype: {x.dtype}")


def as_unsigned(x: TArray) -> TArray:
    """ Convert the signed integer array to be unsigned. """
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.int8:
            return x.to(torch.uint8)
    else:
        if x.dtype == np.int8:
            return x.astype(np.uint8)
        elif x.dtype == np.int16:
            return x.astype(np.uint16)
        elif x.dtype == np.int32:
            return x.astype(np.uint32)
        elif x.dtype == np.int64:
            return x.astype(np.uint64)
        elif x.dtype == np.int128:
            return x.astype(np.uint128)
        elif x.dtype == np.int256:
            return x.astype(np.uint256)

    raise ValueError(f"Unsupported dtype: {x.dtype}")


def as_dtype(x: TArray, dtype: type) -> TArray:
    """ Convert the array to dtype. """
    if isinstance(x, torch.Tensor):
        return x.to(as_torch_dtype(x.dtype))
    else:
        return x.astype(as_np_dtype(dtype))


def numpy_to_tensor(x: np.ndarray, strict: bool = None) -> torch.Tensor:
    """ Convert the numpy array to be Tensor with compatible dtype.

    Args:
        x: numpy array
        strict: if True, raise error if the dtype is incompatible with torch.

    Returns:
        torch.Tensor: The tensor with compatible dtype.
        np.dtype: The dtype of the original numpy array.
    """
    dtype = x.dtype
    if isinstance(x, np.ndarray):
        if dtype == np.uint16:
            if x.max() < 2**15:
                x = x.astype(np.int16)
            else:
                if strict:
                    raise ValueError("Cannot convert uint16 array with values >= 2**15 to tensor."
                                     "Set strict=True to ignore this error.")
                if strict is None:
                    warn("uint16 array with values >= 2**15 will be converted to torch.int32. "
                         "Set strict=True to ignore this warnings.")
                x = x.astype(np.int32)
        elif dtype == np.uint32:
            if x.max() < 2**31:
                x = x.astype(np.int32)
            else:
                if strict:
                    raise ValueError("Cannot convert uint32 array with values >= 2**31 to tensor."
                                     "Set strict=True to ignore this error.")
                if strict is None:
                    warn("uint32 array with values >= 2**31 will be converted to torch.int64. "
                         "Set strict=True to avoid warnings.")
                x = x.astype(np.int64)
        elif dtype == np.uint64:
            if x.max() >= 2**63:
                raise ValueError("Cannot convert uint64 array with values >= 2**63 to tensor")
            x = x.astype(np.int64)

    return torch.as_tensor(x)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """ Convert the tensor to numpy array. """
    return x.detach().cpu().numpy()


def reshape_tensor(x: torch.Tensor, multichannel: bool = True, batch: bool = False) -> torch.Tensor:
    """ Reshape the tensor to be (C, H, W) form.

    Args:
        x: input tensor.
        multichannel: if False, output will be in (H, W) form if C == 1.
        batch: if True, output will be in (B, C, H, W) form.
    """
    assert not (batch and not multichannel), "Cannot reshape batch tensor to (B, H, W) form."

    while x.ndim < 2:
        x = x.unsqueeze(0)

    if x.ndim == 2 and multichannel:
        x = x.unsqueeze(0)
    if x.ndim >= 3 and x.shape[-3] not in [1, 3] and x.shape[-1] in [1, 3]:
        x = x.swapaxes(-1, -2).swapaxes(-2, -3)
    if x.ndim == 3 and x.shape[-3] == 1 and not multichannel:
        x = x.squeeze(-3)

    return x


def reshape_numpy(x: np.ndarray, multichannel: bool = False) -> np.ndarray:
    """ Reshape the numpy array to be (H, W, C) form.

    Args:
        x: input numpy array.
        multichannel: if False, output will be in (H, W) form if C == 1.
    """
    while x.ndim < 2:
        x = x[np.newaxis, ...]

    if x.ndim == 2 and multichannel:
        x = x[..., np.newaxis]
    if x.ndim >= 3 and x.shape[-1] not in [1, 3] and x.shape[-3] in [1, 3]:
        x = x.swapaxes(-3, -2).swapaxes(-2, -1)
    if x.ndim == 3 and x.shape[-1] == 1 and not multichannel:
        x = x.squeeze(-1)

    return x


def im2tensor(x: Union[torch.Tensor, np.ndarray, PIL.Image.Image], multichannel: bool = True) -> torch.Tensor:
    """ Convert image to tensor.

    Args:
        x: input image.
        multichannel: if False, output will be in (H, W) form if C == 1.
    """
    if isinstance(x, PIL.Image.Image):
        x = Ftrans.pil_to_tensor(x)
    elif not isinstance(x, torch.Tensor):
        x = numpy_to_tensor(np.asarray(x))

    x = reshape_tensor(x, multichannel=multichannel)

    return x


def im2numpy(x: Union[torch.Tensor, np.ndarray, PIL.Image.Image], multichannel: bool = False) -> np.ndarray:
    """ Convert image to numpy array.

    Args:
        x: input image.
        multichannel: if False, output will be in (H, W) form if C == 1.
    """
    if isinstance(x, torch.Tensor):
        x = tensor_to_numpy(x)
    else:
        x = np.asarray(x)

    x = reshape_numpy(x, multichannel=multichannel)

    return x


def im2pil(x: Union[torch.Tensor, np.ndarray, PIL.Image.Image]) -> PIL.Image.Image:
    """ Convert image to PIL.Image. """
    if isinstance(x, PIL.Image.Image):
        return x
    if any(isinstance(xx, PIL.Image.Image) for xx in x):
        return [PIL.Image.fromarray(xx) for xx in x]
    if any(isinstance(xxx, PIL.Image.Image) for xx in x for xxx in xx):
        return [[PIL.Image.fromarray(xxx) for xxx in xx] for xx in x]

    x = im2numpy(x)

    if x.ndim == 5 or (x.ndim == 4 and x.shape[-1] not in [1, 3]):
        return [[PIL.Image.fromarray(xxx) for xxx in xx] for xx in x]
    if x.ndim == 4 or (x.ndim == 3 and x.shape[-1] not in [1, 3]):
        return [PIL.Image.fromarray(xx) for xx in x]

    return PIL.Image.fromarray(x)


def normalize(
    x: TArray,
    black_level: Union[int, list[int]],
    white_level: int,
    channel_axis: int = None,
    out_dtype: type = np.float32,
) -> TArray:
    """ Normalize image to [0, 1].

    Args:
        x: Input image.
        black_level: The black level of the input image, as scalar or per-channel vector (channel_axis required).
        white_level: The white level of the input image, as scalar.
        out_dtype: The output data type, should be floating types.
        channel_axis: The channel axis of the image.
    """
    assert is_integer_dtype(x.dtype), "Input image must be integer type."
    assert is_floating_dtype(out_dtype), "Output data type must be floating type."

    # parameter check
    if isinstance(x, torch.Tensor):
        black_level = torch.as_tensor(black_level) if black_level is not None else 0
        white_level = torch.as_tensor(white_level) if white_level is not None else torch.iinfo(x.dtype).max

    else:
        black_level = np.asarray(black_level) if black_level is not None else 0
        white_level = np.asarray(white_level) if white_level is not None else np.iinfo(x.dtype).max

    # prevent underflow
    if x.min() < black_level:
        x = as_signed(x)

    # normalize
    if channel_axis is None:
        x = (x - black_level) / (white_level - black_level)
    else:
        assert x.shape[channel_axis] == len(black_level), \
            "black_level must have the same length as the number of channels"

        x = x.swapaxes(channel_axis, -1)
        x = (x - black_level) / (white_level - black_level)
        x = x.swapaxes(-1, channel_axis)

    return as_dtype(x, out_dtype)


def denormalize(
    x: TArray,
    black_level: Union[int, list[int]],
    white_level: int,
    channel_axis: int = None,
    out_dtype: type = np.uint8,
) -> TArray:
    """ Denormalize image from [0, 1] to [black_level, white_level].

    Args:
        x: Input image.
        black_level: The black level of the output image, as scalar or per-channel vector (channel_axis required).
        white_level: The white level of the output image, as scalar.
        out_dtype: The output data type, should be integer types.
        channel_axis: The channel axis of the image.
    """
    assert is_floating_dtype(x.dtype), "Input image must be floating type."
    assert is_integer_dtype(out_dtype), "Output data type must be integer type."

    # parameter check
    if isinstance(x, torch.Tensor):
        black_level = torch.as_tensor(black_level) if black_level is not None else 0
        white_level = torch.as_tensor(white_level) if white_level is not None else torch.iinfo(out_dtype).max

    else:
        black_level = np.asarray(black_level) if black_level is not None else 0
        white_level = np.asarray(white_level) if white_level is not None else np.iinfo(out_dtype).max

    # denormalize
    if channel_axis is None:
        x = x * (white_level - black_level) + black_level
    else:
        assert x.shape[channel_axis] == len(black_level), \
            "black_level must have the same length as the number of channels"

        x = x.swapaxes(channel_axis, -1)
        x = x * (white_level - black_level) + black_level
        x = x.swapaxes(-1, channel_axis)

    out_dtype = np.dtype(out_dtype)
    x = x.clip(np.iinfo(out_dtype).min, np.iinfo(out_dtype).max).round()

    return as_dtype(x, out_dtype)


def _bit_length_of_dtype(dtype: type) -> int:
    max_value = np.iinfo(as_np_dtype(dtype)).max
    return max_value.bit_length()


def im2float(x: TArray, in_max_value: int = None, in_bit_length: int = None, out_dtype: type = np.float32) -> TArray:
    """ Convert image to float.

    If the input image is floating type, the output image is the same.
    If the input image is integer type, the output image is normalized to [0, 1].
    Set in_max_value or in_bit_length to override the maximum value of the input integer image.

    Args:
        x: image tensor
        in_max_value: the maximum value of the input image
        in_bit_length: bit length of the input if x is integer type
        out_dtype: output dtype
    """
    assert is_floating_dtype(out_dtype), "output data type must be floating type"
    assert in_max_value is None or in_bit_length is None, \
        "in_max_value and in_bit_length cannot be specified at the same time"

    if is_floating_dtype(x.dtype):
        return as_dtype(x, out_dtype)

    if is_integer_dtype(x.dtype):
        if in_max_value is None:
            in_bit_length = in_bit_length or _bit_length_of_dtype(x.dtype)
            in_max_value = 2 ** in_bit_length - 1

        return normalize(x, 0, in_max_value, out_dtype=out_dtype)

    raise TypeError(f"unsupported array type: {x.dtype}")


def im2double(x: TArray, in_max_value: int = None, in_bit_length: int = None, out_dtype: type = np.float64) -> TArray:
    """ Convert image to double. See `im2float` for details. """
    return im2float(x, in_max_value=in_max_value, in_bit_length=in_bit_length, out_dtype=out_dtype)


def im2uint8(
    x: TArray,
    in_max_value: int = None,
    out_max_value: int = None,
    in_bit_length: int = None,
    out_bit_length: int = None,
    out_dtype: type = np.uint8,
) -> TArray:
    """ Convert image to uint8.

    If the input image is floating type, the output image is quantized to [0, out_max_value].
    Set out_max_value or out_bit_length to override the maximum value of the output integer image.
    If the input image is integer type, it is normalized to floating type before quantization.
    Set in_max_value or in_bit_length to override the maximum value of the input integer image.

    Args:
        x: image tensor
        in_max_value: the maximum value of the input image
        out_max_value: the maximum value of the output image
        in_bit_length: bit length of the input if x is integer type
        out_bit_length: bit length of the output if x is integer type
        out_dtype: output dtype
    """
    if in_bit_length is None and out_bit_length is None and as_np_dtype(x.dtype) == as_np_dtype(out_dtype):
        return x

    if out_max_value is None:
        out_bit_length = out_bit_length or _bit_length_of_dtype(out_dtype)
        out_max_value = 2 ** out_bit_length - 1

    if is_integer_dtype(x.dtype):
        if in_max_value is None:
            in_bit_length = in_bit_length or _bit_length_of_dtype(x.dtype)
            in_max_value = 2 ** in_bit_length - 1

        if in_max_value == out_max_value:
            return as_dtype(x, out_dtype)

        if x.dtype in [np.uint64, np.uint128, np.int64, np.int128, torch.int64]:
            mid_dtype = np.float128
        elif x.dtype in [np.uint32, np.int32, torch.int32]:
            mid_dtype = np.float64
        else:
            mid_dtype = np.float32

        x = im2float(x, in_max_value=in_max_value, out_dtype=mid_dtype)

    return denormalize(x, 0, out_max_value, out_dtype=out_dtype)


def im2uint16(
    x: TArray,
    in_max_value: int = None,
    out_max_value: int = None,
    in_bit_length: int = None,
    out_bit_length: int = None,
    out_dtype: type = np.uint16,
) -> TArray:
    """ Convert image to uint16. See `im2uint8` for details. """
    return im2uint8(
        x,
        in_max_value=in_max_value,
        out_max_value=out_max_value,
        in_bit_length=in_bit_length,
        out_bit_length=out_bit_length,
        out_dtype=out_dtype
    )
