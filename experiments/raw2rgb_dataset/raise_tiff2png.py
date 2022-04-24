from pathlib import Path
import argparse
import csv
import multiprocessing as mp

import wget
from tqdm.auto import tqdm

import utils

parser = argparse.ArgumentParser("Download RAISE dataset")
parser.add_argument("-o", "--data_path", default="data/RAISE")
args = parser.parse_args()


def convert(i, tif_path: Path):
    try:
        tif = utils.loadtiff(tif_path)
        tif = utils.im2float(tif)
        tif = utils.im2uint8(tif)
        png_path = Path(args.data_path, tif_path.relative_to(args.data_path).as_posix().replace("TIFF", "PNG"))
        utils.imsave(tif, png_path)
        jpg_path = Path(args.data_path, tif_path.relative_to(args.data_path).as_posix().replace("TIFF", "JPG"))
        utils.imsave(tif, jpg_path, quality=95)
    except OSError as e:
        print(e)


with mp.Pool(mp.cpu_count()) as pool:
    tiffs = Path(args.data_path).glob("**/*.TIFF")
    results = [pool.apply_async(convert, row) for row in enumerate(tiffs)]
    for res in tqdm(results):
        res.get()
