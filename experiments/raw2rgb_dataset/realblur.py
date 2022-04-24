from pathlib import Path
import argparse
import re
import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm

import utils


parser = argparse.ArgumentParser("Generate RealBlur RAW-RGB dataset")
parser.add_argument("src_dir", type=str, help="Source directory")
parser.add_argument("dst_dir", type=str, help="Destination directory")
args = parser.parse_args()


def get_datalist(data_dir, data_type):
    for raw_path in Path(data_dir).glob(f"**/{data_type}*.ARW"):
        jpg_path = raw_path.with_suffix(".JPG")

        rel_path = raw_path.relative_to(data_dir).as_posix()
        date, name = re.fullmatch(r"^(.+?)\/.+?\/(.+?)\..+?$", rel_path).groups()

        yield raw_path, jpg_path, date, name


srgbd50_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
srgbd65_to_xyz = np.array([[0.4360747, 0.3850649, 0.1430804],
                           [0.2225045, 0.7168786, 0.0606169],
                           [0.0139322, 0.0971045, 0.7141733]])
d50_to_d65 = srgbd50_to_xyz @ np.linalg.inv(srgbd65_to_xyz)
d65_to_d50 = np.linalg.inv(d50_to_d65)


def load_images(raw_path, jpg_path, data_type):
    raw_file = utils.loadraw(raw_path)
    raw: np.ndarray = raw_file.raw_image_visible.copy()[8:-8, 8:-8]

    pattern = "".join("RGBG"[i] for i in raw_file.raw_pattern.flatten())

    wb = np.array(raw_file.camera_whitebalance[:3])

    ccm = np.array(raw_file.rgb_xyz_matrix[:3, :3])
    ccm = ccm / np.sum(ccm, axis=1, keepdims=True)
    ccm = d50_to_d65 @ np.linalg.inv(ccm)

    white = raw_file.white_level
    black = raw_file.black_level_per_channel[:3]

    jpg = utils.im2numpy(utils.imload(jpg_path))

    if data_type == "CLN":
        T, L, H, W = (0, 1400, 3150, 2840)
    elif data_type == "BLR":
        T, L, H, W = (0, 870, 3110, 2820)

    raw = raw[T:T + H, L:L + W]
    jpg = jpg[T:T + H, L:L + W]

    return raw, jpg, pattern, wb, ccm, black, white


if __name__ == "__main__":
    a7r3ccm = utils.loadmat(utils.path.get("data/a7r3_cam2xyz.mat"))["final_cam2xyz"]

    for data_type in ["CLN", "BLR"]:
        datalist = list(get_datalist(args.src_dir, data_type))
        print(f"{data_type}: {len(datalist)}")

        Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{args.dst_dir}/{data_type}.txt", "w") as f:
            for raw_path, jpg_path, date, name in tqdm(datalist, desc="datalist"):
                f.write(f"{date}/{data_type}/{name}.tiff {date}/{data_type}/{name}.png\n")

        def gen_raw_rgb(raw_path, jpg_path, date, name):
            raw, jpg, pattern, wb, ccm, black, white = load_images(raw_path, jpg_path, data_type)
            Path(f"{args.dst_dir}/{date}/{data_type}").mkdir(parents=True, exist_ok=True)
            utils.savetiff(raw, f"{args.dst_dir}/{date}/{data_type}/{name}.tiff", metadata={
                "bayer_pattern": pattern,
                "white_balance": wb.tolist(),
                "color_matrix": ccm.tolist(),
                "color_matrix_adjusted": a7r3ccm.tolist(),
                "black_level": black,
                "white_level": white,
            })
            utils.imsave(jpg, f"{args.dst_dir}/{date}/{data_type}/{name}.png")

        with mp.Pool(mp.cpu_count()) as pool:
            results = [pool.apply_async(gen_raw_rgb, row) for row in datalist]
            for res in tqdm(results, desc="images"):
                res.get()
