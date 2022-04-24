from pathlib import Path
import argparse
import csv
import multiprocessing as mp

import wget
from tqdm.auto import tqdm

import utils

parser = argparse.ArgumentParser("Download RAISE dataset")
parser.add_argument("-f", "--csv_path", default="data/datalist/RAISE_all.csv")
parser.add_argument("-o", "--out_path", type=str, default="data/RAISE")
args = parser.parse_args()

with open(args.csv_path) as f:
    reader = csv.reader(f)

    header = next(reader)
    Path(f"{args.out_path}/{header[1]}").mkdir(parents=True, exist_ok=True)
    Path(f"{args.out_path}/{header[2]}").mkdir(parents=True, exist_ok=True)

    def dl_inst(i, row):
        if i == 0:
            return
        wget.download(row[1], f"{args.out_path}/{header[1]}/{row[0]}.{header[1]}", bar=lambda *a: None)
        wget.download(row[2], f"{args.out_path}/{header[2]}/{row[0]}.{header[2]}", bar=lambda *a: None)
        return i, row[0], row[1], row[2]

    with mp.Pool(mp.cpu_count()) as pool:
        results = [pool.apply_async(dl_inst, (i, row)) for i, row in enumerate(reader)]
        for res in tqdm(results):
            print(res.get())
