import torch
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pad
from torchvision.utils import make_grid
# from skimage.util import montage
from toolz import isiterable

from .convert import im2numpy, im2tensor, im2float, im2uint8


def _enumerate_noneable(iterable):
    if iterable is None:
        return []
    return enumerate(iterable)


def imgrid(images, padding=2, plot=False) -> np.ndarray:
    """ Montage images. """
    assert isiterable(images) and len(images) > 0 and all(map(isiterable, images)), "images must be a 2D list of images"
    assert any(len(row) > 0 for row in images), "images must contains at least one image"

    nrows = len(images)
    ncols = max(len(row) for row in images)

    imgs = [torch.zeros(3, 1, 1, dtype=torch.uint8)
            for _ in range(ncols)
            for _ in range(nrows)]

    max_h, max_w = 1, 1
    for R, row in _enumerate_noneable(images):
        for C, img in _enumerate_noneable(row):
            img = im2uint8(im2tensor(img))
            assert img.ndim == 3, f"image must be a 3D tensor, received {img.ndim}"
            if img.shape[0] == 1:
                img = img.expand(3, *img.shape[1:])

            h, w = img.shape[-3:-1]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

            imgs[R * ncols + C] = img

    for i, img in enumerate(imgs):
        h, w = img.shape[-3:-1]
        imgs[i] = pad(img, (0, 0, max_w - w, max_h - h), padding_mode="constant", fill=0)

    results = make_grid(imgs, nrow=ncols, padding=padding)
    if plot:
        from IPython.display import display
        display(PIL.Image.fromarray(results.permute(1, 2, 0).numpy()))

    return results


def imgridplot(images, *, figsize=None,
               suptitle=None, supxlabel=None, supylabel=None, suptitlesize=None, suplabelsize=None,
               titles=None, xlabels=None, ylabels=None, titlesize=None, labelsize=None,
               xtick=False, ytick=False, border=True) -> tuple[plt.Figure, plt.Axes]:
    """ Plot images.

    Args:
        images: PIL.Image, numpy.ndarray, or torch.Tensor. (1, C, H, W), (C, H, W), (H, W, C) or (H, W).
        suptitle: title of entire figure. `supxlabel` and `supylabel` has same rules.
        titles: title of each subplot. `xlabels` and `ylabels` has same rules.
        xtick: enable axis ticks. `ytick` has same rules.
        border: show borders around of suplots.
    """
    assert not (not border and (xtick or ytick)), "`xtick` and `ytick` cannot be True when `border` is False."
    assert isiterable(images) and len(images) > 0 and all(map(isiterable, images)), "images must be a 2D list of images"
    assert any(len(row) > 0 for row in images), "images must contains at least one image"

    nrows = len(images)
    ncols = max(len(row) for row in images)

    fig, ax = plt.subplots(nrows, ncols, squeeze=False,
                           dpi=300, figsize=figsize,
                           gridspec_kw={"wspace": 0.01, "hspace": 0.01},
                           constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    for ax_row in ax:
        for ax_elem in ax_row:
            if not border:
                ax_elem.axis("off")
            else:
                if not xtick:
                    ax_elem.set_xticks([])
                if not ytick:
                    ax_elem.set_yticks([])

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitlesize)
    if supxlabel:
        fig.supxlabel(supxlabel, fontsize=suplabelsize)
    if supylabel:
        fig.supylabel(supylabel, fontsize=suplabelsize)

    elems = [[
        {"img": None, "title": None, "xlabel": None, "ylabel": None}
        for _ in range(ncols)]
        for _ in range(nrows)]

    for R, row in _enumerate_noneable(images):
        for C, img in _enumerate_noneable(row):
            elems[R][C]["img"] = im2float(im2numpy(img)).squeeze()

    for R, row in _enumerate_noneable(titles):
        for C, title in _enumerate_noneable(row):
            elems[R][C]["title"] = title

    for R, row in _enumerate_noneable(xlabels):
        for C, xlabel in _enumerate_noneable(row):
            elems[R][C]["xlabel"] = xlabel

    for R, row in _enumerate_noneable(ylabels):
        for C, ylabel in _enumerate_noneable(row):
            elems[R][C]["ylabel"] = ylabel

    for R, row in enumerate(elems):
        for C, elem in enumerate(row):
            if elem["img"] is None:
                ax[R][C].set_visible(False)
            else:
                ax[R][C].imshow(elem["img"])
                if elem["title"] is not None:
                    ax[R][C].set_title(elem["title"], fontsize=titlesize)
                if elem["xlabel"] is not None:
                    ax[R][C].set_xlabel(elem["xlabel"], fontsize=labelsize)
                if elem["ylabel"] is not None:
                    ax[R][C].set_ylabel(elem["ylabel"], fontsize=labelsize)

    return fig, ax
