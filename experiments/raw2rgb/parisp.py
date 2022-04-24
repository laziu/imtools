#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import itertools
import os
import multiprocessing as mp
import gc
from datetime import datetime
from warnings import warn
from pprint import pprint
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms
import torchvision.datasets
import torchvision.models
import torchmetrics.functional
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets.raw
import datasets.raw.cached
import models.paramisp
import utils
import utils.color
import utils.bayer

os.environ["LRU_CACHE_SIZE"] = "16"
utils.env.load()


parser = argparse.ArgumentParser(description="RAW-RGB + Parameterized ISP",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("mode", choices=["train", "test"], help="train or test")
parser.add_argument("-o", "--save-name", default="RAW2RGB_ParISP", type=str, metavar="NAME",
                    help="name to save results")
parser.add_argument("-O", "--save-root", default=utils.path.get("results"), type=str, metavar="PATH",
                    help="parent directory to save results")
parser.add_argument("--data-rbr", "--data-root-realblurraw", metavar="PATH",
                    default=utils.env.get("REALBLUR_RAW"),
                    help="path to the RealBlur RAW dataset")
parser.add_argument("--data-raise", "--data-root-raise", metavar="PATH",
                    default=utils.env.get("RAISE"),
                    help="path to the RAISE dataset")
parser.add_argument("--list-rbr-train", "--data-list-realblurraw-train", metavar="PATH",
                    default=utils.path.get("data/datalist/realblur_raw_all_train.csv"),
                    help="datalist of the RealBlur RAW training dataset")
parser.add_argument("--list-rbr-test", "--data-list-realblurraw-test", metavar="PATH",
                    default=utils.path.get("data/datalist/realblur_raw_all_test.csv"),
                    help="datalist of the RealBlur RAW test dataset")
parser.add_argument("--list-raise-train", "--data-list-raise-train", metavar="PATH",
                    default=utils.path.get("data/datalist/raise_raw_train.csv"),
                    help="datalist of the RAISE training dataset")
parser.add_argument("--list-raise-test", "--data-list-raise-test", metavar="PATH",
                    default=utils.path.get("data/datalist/raise_raw_test.csv"),
                    help="datalist of the RAISE test dataset")
parser.add_argument("--data-cache", metavar="PATH",
                    default=utils.env.get("DATA_CACHE"),
                    help="path to cache the splitted dataset")
parser.add_argument("--crop-size", default=512, type=int, metavar="N",
                    help="crop size of the input image")
parser.add_argument("--max-epochs", "--epochs", default=300, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=4, type=int, metavar="N",
                    help="batch size")
parser.add_argument("--lr", default=1e-4, type=float, metavar="LR",
                    help="learning rate")
parser.add_argument("--lr-step", default=20, type=int, nargs="+",
                    help="scheduler step")
parser.add_argument("--lr-gamma", default=0.9, type=float, metavar="G",
                    help="scheduler gamma")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-8, type=float, metavar="W",
                    help="weight decay")
parser.add_argument("-j", "--data-workers", type=int, metavar="N",
                    default=mp.cpu_count() // (2 * torch.cuda.device_count()),
                    help="number of data loading workers per GPU")
parser.add_argument("--seed", default=None, type=int,
                    help="random seed for initializing training")
parser.add_argument("--resume", nargs="?", const="auto", default=None, type=str, metavar="PATH",
                    help="path to latest checkpoint")
parser.add_argument("--pretrained", default=None, nargs=2, type=str, metavar="PATH",
                    help="path to pretrained model: sequence of raw2rgb, rgb2raw")
parser.add_argument("--ckpt-freq", default=50, type=int, metavar="N",
                    help="save checkpoint every N epochs")
parser.add_argument("--ckpt-topk", default=3, type=int, metavar="N",
                    help="save only last k checkpoints")
parser.add_argument("--profiler", choices=["simple", "advanced", "pytorch"], default=None,
                    help="profiler to use")
args = parser.parse_args()

if not hasattr(args, "timestamp"):
    args.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

args.save_dir = Path(args.save_root, args.save_name).as_posix()
args.ckpt_dir = f"{args.save_dir}/checkpoints"
args.wght_dir = f"{args.save_dir}/weights"
args.tlog_dir = f"{args.save_dir}/train/{args.timestamp}"

if args.resume == "auto":
    args.resume = f"{args.ckpt_dir}/last.ckpt"
args.resume = utils.path.purify(args.resume)

if args.pretrained is not None:
    args.pretrained = [utils.path.purify(p) for p in args.pretrained]

args.gpus = torch.cuda.device_count()
args.deterministic = args.seed is not None


def train():
    pprint(args)

    Path(args.wght_dir).mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        pl.seed_everything(args.seed)
        warn("You have chosen to seed training. "
             "This will turn on the CUDNN deterministic setting, which can slow down your training considerably! "
             "You may see unexpected behavior when restarting from checkpoints.")

    data = Data(**vars(args))
    model = Model(**vars(args))

    if args.pretrained is not None:
        model.raw2rgb.load_state_dict(torch.load(args.pretrained[0]))
        model.rgb2raw.load_state_dict(torch.load(args.pretrained[1]))

    print(f"Logging to {args.save_dir}/train/{args.timestamp}")
    logger = TensorBoardLogger(save_dir=args.save_dir, name="train", version=args.timestamp)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir, monitor="val/psnr/rgb", mode="max", auto_insert_metric_name=False,
            save_top_k=args.ckpt_topk, save_last=True,
            filename=f"psnr={{val/psnr:.4f}}_{args.timestamp}_{{epoch:03d}}"),
        ModelCheckpoint(
            dirpath=args.ckpt_dir, save_top_k=-1, every_n_epochs=args.ckpt_freq, auto_insert_metric_name=False,
            filename=f"{args.timestamp}_{{epoch:03d}}_psnr={{val/psnr:.4f}}"),
    ]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, profiler=args.profiler)

    trainer.fit(model, datamodule=data, ckpt_path=args.resume)

    trainer.save_checkpoint(f"{args.ckpt_dir}/final_{args.timestamp}.ckpt")
    torch.save(model.raw2rgb.state_dict(), f"{args.wght_dir}/raw2rgb_{args.timestamp}.pt")
    torch.save(model.rgb2raw.state_dict(), f"{args.wght_dir}/rgb2raw_{args.timestamp}.pt")


class Data(pl.LightningDataModule):
    def __init__(self, batch_size, data_workers,
                 data_rbr,   list_rbr_train,   list_rbr_test,
                 data_raise, list_raise_train, list_raise_test,
                 data_cache, crop_size, **kwds):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        self.rbr_train = datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_train, cache_dir=self.hparams.data_cache + "/realblur_raw",
            crop_config=datasets.raw.cached.RandomCropConfig(self.hparams.crop_size + 5), unify_bayer=True)
        self.rbr_val   = datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_test,  cache_dir=self.hparams.data_cache + "/realblur_raw",
            crop_config=datasets.raw.cached.CenterCropConfig(self.hparams.crop_size + 5), unify_bayer=True)
        self.rbr_test  = datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_test,  cache_dir=self.hparams.data_cache + "/realblur_raw",
            unify_bayer=True)
        self.raise_train = datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_train, cache_dir=self.hparams.data_cache + "/raise_raw",
            crop_config=datasets.raw.cached.RandomCropConfig(self.hparams.crop_size + 5), unify_bayer=True)
        self.raise_val   = datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_test,  cache_dir=self.hparams.data_cache + "/raise_raw",
            crop_config=datasets.raw.cached.CenterCropConfig(self.hparams.crop_size + 5), unify_bayer=True)
        self.raise_test  = datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_test,  cache_dir=self.hparams.data_cache + "/raise_raw",
            unify_bayer=True)
        self.trainset = datasets.raw.Concatenate(self.rbr_train, self.raise_train)
        self.valset   = datasets.raw.Concatenate(self.rbr_val,   self.raise_val)
        self.testset  = datasets.raw.Concatenate(self.rbr_test,  self.raise_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size, shuffle=True, drop_last=True,
            num_workers=self.hparams.data_workers, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size, shuffle=False, drop_last=True,
            num_workers=self.hparams.data_workers, pin_memory=False, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.hparams.data_workers)


SRGB_D50_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
SRGB_D65_TO_XYZ = np.array([[0.4360747, 0.3850649, 0.1430804],
                            [0.2225045, 0.7168786, 0.0606169],
                            [0.0139322, 0.0971045, 0.7141733]])
XYZ_TO_SRGB_D65 = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.969266,  1.8760108,  0.0415560],
                            [0.0556434, -0.2040259,  1.0572252]])
D50_TO_D65 = SRGB_D50_TO_XYZ @ np.linalg.inv(SRGB_D65_TO_XYZ)
D65_TO_D50 = np.linalg.inv(D50_TO_D65)
D50_TO_D65 = torch.as_tensor(D50_TO_D65, dtype=torch.float32).unsqueeze(0)
D65_TO_D50 = torch.as_tensor(D65_TO_D50, dtype=torch.float32).unsqueeze(0)
XYZ_TO_SRGB_D65 = torch.as_tensor(XYZ_TO_SRGB_D65, dtype=torch.float32).unsqueeze(0)


class Model(pl.LightningModule):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, tlog_dir, **kwds):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.raw2rgb = models.paramisp.BasicPipeline()
        self.rgb2raw = models.paramisp.BasicPipeline()

        self.whitebalance = utils.color.WhiteBalance()
        self.demosaic     = utils.bayer.Demosaic5x5()
        self.mosaic       = utils.bayer.Mosaic()
        self.colormatrix  = utils.color.ColorMatrix()
        self.xyz2rgb      = utils.color.XYZ2RGB()
        self.rgb2xyz      = utils.color.RGB2XYZ()
        self.linear2srgb  = utils.color.Linear2SRGB()
        self.srgb2linear  = utils.color.SRGB2Linear()

        self.register_buffer("M_d652d50", D65_TO_D50)
        self.register_buffer("M_xyz2rgb", XYZ_TO_SRGB_D65)

    def destruct_batch(self, batch: dict):
        raw: torch.Tensor = batch["raw"]  # B, 1, H, W
        rgb: torch.Tensor = batch["rgb"]  # B, 3, H, W
        bayer_mask:    torch.Tensor = batch["bayer_mask"].unsqueeze(1).to(torch.int64)    # B, 1, H, W
        white_balance: torch.Tensor = batch["white_balance"].unsqueeze(-1).unsqueeze(-1)  # B, 3, 1, 1
        color_matrix:  torch.Tensor = self.M_d652d50 @ batch["color_matrix"]  # B, 3, 3
        camera_ids: list[str] = batch["camera_id"]  # B
        names: list[str] = batch["name"]  # B

        raw = raw[..., 2:-2, 2:-2]
        rgb = rgb[..., 2:-2, 2:-2]
        bayer_mask = bayer_mask[..., 2:-2, 2:-2]

        color_matrix = self.M_xyz2rgb @ color_matrix
        color_matrix = color_matrix / torch.mean(color_matrix, dim=-1, keepdim=True)

        return raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names

    def process_raw2mid(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.demosaic(y, bayer_mask)
        y = self.colormatrix(y, color_matrix)
        return y

    def process_rgb2mid(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.srgb2linear(y)
        return y

    def process_raw2rgb(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.demosaic(y, bayer_mask)
        y = self.colormatrix(y, color_matrix)
        y = self.raw2rgb(y)
        y = self.linear2srgb(y)
        return y

    def process_rgb2raw(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.srgb2linear(y)
        y = self.rgb2raw(y)
        y = self.colormatrix(y, torch.inverse(color_matrix))
        y = self.mosaic(y, bayer_mask)
        y = self.whitebalance(y, 1 / white_balance, bayer_mask)
        return y

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_con = F.l1_loss(x, y)
        loss_fft = F.l1_loss(torch.fft.fft2(x, norm="backward"),
                             torch.fft.fft2(y, norm="backward"))
        return loss_con + 0.1 * loss_fft

    def training_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)
        optim_raw2rgb, optim_rgb2raw = self.optimizers()

        raw_mid = self.process_raw2mid(raw, bayer_mask, white_balance, color_matrix)
        rgb_mid = self.process_rgb2mid(rgb, bayer_mask, white_balance, color_matrix)

        rgb_mid_est = self.raw2rgb(raw_mid)
        loss_rgb = self.compute_loss(rgb_mid_est, rgb_mid)
        # rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix)
        # loss_rgb = self.compute_loss(rgb_est, rgb)

        optim_raw2rgb.zero_grad()
        self.manual_backward(loss_rgb)
        optim_raw2rgb.step()

        raw_mid_est = self.rgb2raw(rgb_mid)
        loss_raw = self.compute_loss(raw_mid_est, raw_mid)
        # raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix)
        # loss_raw = self.compute_loss(raw_est, raw)

        optim_rgb2raw.zero_grad()
        self.manual_backward(loss_raw)
        optim_rgb2raw.step()

        self.log_dict({
            "train/loss/raw": loss_raw,
            "train/loss/rgb": loss_rgb,
        }, prog_bar=True)

        self.log_dict({
            "step": self.current_epoch,
            "train/epoch/loss/raw": loss_raw,
            "train/epoch/loss/rgb": loss_rgb,
        }, on_step=False, on_epoch=True)

        # if self.trainer.is_last_batch:
        # return loss

    def training_epoch_end(self, outputs):
        # super().training_epoch_end(outputs)
        sched_raw2rgb, sched_rgb2raw = self.lr_schedulers()
        sched_raw2rgb.step()
        sched_rgb2raw.step()

        # gc_target = np.asarray(gc.get_count())
        # gc.collect()
        # torch.cuda.empty_cache()
        # gc_del = np.asarray(gc.get_count()) - gc_target

        # if gc_del[0] > 0 or gc_del[1] > 0 or gc_del[2] > 0:
        #     print(f"Garbage collected {gc_del} objects")

        # return super().training_epoch_end(outputs)

    def compute_metrics(self, x: torch.Tensor, y: torch.Tensor):
        x = x.clip(0, 1)
        y = y.clip(0, 1)
        return {
            "loss":  self.compute_loss(x, y),
            "psnr":  torchmetrics.functional.peak_signal_noise_ratio(x, y, data_range=1.),
            "ssim":  torchmetrics.functional.structural_similarity_index_measure(x, y, data_range=1.),
            "mssim": torchmetrics.functional.multiscale_structural_similarity_index_measure(x, y, data_range=1.),
        }

    def validation_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix)
        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix)

        met_rgb = self.compute_metrics(rgb_est, rgb)
        met_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "val/loss/rgb":  met_rgb["loss"],
            "val/psnr/rgb":  met_rgb["psnr"],
            "val/ssim/rgb":  met_rgb["ssim"],
            "val/mssim/rgb": met_rgb["mssim"],
            "val/loss/raw":  met_raw["loss"],
            "val/psnr/raw":  met_raw["psnr"],
            "val/ssim/raw":  met_raw["ssim"],
            "val/mssim/raw": met_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if batch_idx == 0:
            plot = utils.imgrid([raw, raw_est, rgb, rgb_est])
            utils.imsave(plot, f"{self.hparams.tlog_dir}/val_latest.png")

    def configure_optimizers(self):
        optim_raw2rgb = torch.optim.Adam(self.raw2rgb.parameters(), lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        optim_rgb2raw = torch.optim.Adam(self.rgb2raw.parameters(), lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        sched_raw2rgb = torch.optim.lr_scheduler.StepLR(
            optim_raw2rgb, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        sched_rgb2raw = torch.optim.lr_scheduler.StepLR(
            optim_rgb2raw, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        return [optim_raw2rgb, optim_rgb2raw], [sched_raw2rgb, sched_rgb2raw]


if __name__ == "__main__":
    if args.mode == "train":
        train()
    raise NotImplementedError()
