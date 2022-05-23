#!/usr/bin/env python
import argparse
from pathlib import Path
import shutil
import random
import itertools
import os
import json
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
parser.add_argument("model_type", choices=["cyisp", "cyisp_wb",
                                           "parisp_indep", "parisp_indep_detach",
                                           "parisp_naive", "parisp_naive_indep",
                                           "parisp_hyper", "parisp_hyper_indep"],
                    help="model type")
parser.add_argument("dataset_type", choices=["all", "realblurraw", "raise"],
                    help="dataset to use")
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
                    default=utils.path.get("data/datalist/realblur_raw_all.train.csv"),
                    help="datalist of the RealBlur RAW training dataset")
parser.add_argument("--list-rbr-val", "--data-list-realblurraw-val", metavar="PATH",
                    default=utils.path.get("data/datalist/realblur_raw_all.val.csv"),
                    help="datalist of the RealBlur RAW validation dataset")
parser.add_argument("--list-rbr-test", "--data-list-realblurraw-test", metavar="PATH",
                    default=utils.path.get("data/datalist/realblur_raw_all.test.csv"),
                    help="datalist of the RealBlur RAW test dataset")
parser.add_argument("--list-raise-train", "--data-list-raise-train", metavar="PATH",
                    default=utils.path.get("data/datalist/raise_raw_all.train.csv"),
                    help="datalist of the RAISE training dataset")
parser.add_argument("--list-raise-val", "--data-list-raise-val", metavar="PATH",
                    default=utils.path.get("data/datalist/raise_raw_all.val.csv"),
                    help="datalist of the RAISE validation dataset")
parser.add_argument("--list-raise-test", "--data-list-raise-test", metavar="PATH",
                    default=utils.path.get("data/datalist/raise_raw_all.test.csv"),
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
                    default=4,  # mp.cpu_count() // (2 * torch.cuda.device_count()),
                    help="number of data loading workers per GPU")
parser.add_argument("--seed", default=None, type=int,
                    help="random seed for initializing training")
parser.add_argument("--resume", nargs="?", const="auto", default=None, type=str, metavar="PATH",
                    help="path of latest results to resume if exists")
parser.add_argument("--resume-ckpt", default=None, type=str, metavar="PATH",
                    help="path of checkpoint to resume if exists; overrides --resume")
parser.add_argument("--resume-logs", default=None, type=str, metavar="PATH",
                    help="path of logs to resume if exists; overrides --resume")
parser.add_argument("--pretrained", default=None, nargs=2, type=str, metavar="PATH",
                    help="path to pretrained model: sequence of raw2rgb, rgb2raw")
parser.add_argument("--ckpt-freq", default=50, type=int, metavar="N",
                    help="save checkpoint every N epochs")
parser.add_argument("--ckpt-topk", default=3, type=int, metavar="N",
                    help="save only last k checkpoints")
parser.add_argument("--profiler", choices=["simple", "advanced", "pytorch"], default=None,
                    help="profiler to use")
parser.add_argument("--save-image", action="store_true", default=False,
                    help="save image when testing")
args = parser.parse_args()

if not hasattr(args, "timestamp"):
    args.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

args.save_dir = Path(args.save_root, args.save_name, args.timestamp).as_posix()
args.postfix_ckpt = "checkpoints"
args.postfix_wght = "weights"
args.postfix_logs = "logs"

if args.pretrained is not None:
    args.pretrained = [utils.path.purify(p) for p in args.pretrained]

args.gpus = torch.cuda.device_count()
args.deterministic = args.seed is not None

if any([args.resume, args.resume_ckpt, args.resume_logs]):
    if args.resume == "auto":
        args.resume = args.save_dir

    if args.resume:
        args.resume = utils.path.purify(args.resume)

        if not args.resume_ckpt:
            args.resume_ckpt = f"{args.resume}/{args.postfix_ckpt}/last.ckpt"

        if not args.resume_logs:
            args.resume_logs = f"{args.resume}/{args.postfix_logs}"


def main():
    pprint(args)

    args.ckpt_dir = f"{args.save_dir}/{args.postfix_ckpt}"
    args.wght_dir = f"{args.save_dir}/{args.postfix_wght}"
    args.logs_dir = f"{args.save_dir}/{args.postfix_logs}/{args.mode}"

    if args.resume_logs:
        shutil.copytree(args.resume_logs, args.logs_dir)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        pl.seed_everything(args.seed)
        warn("You have chosen to seed training. "
             "This will turn on the CUDNN deterministic setting, which can slow down your training considerably! "
             "You may see unexpected behavior when restarting from checkpoints.")

    match args.dataset_type:
        case "realblurraw": data = Data_RealBlurRaw
        case "raise":       data = Data_Raise
        case "all":         data = Data_All
    data = data(**vars(args))

    match args.model_type:
        case "cyisp":               model = Model_Cyisp
        case "cyisp_wb":            model = Model_CyispWB
        case "parisp_indep":        model = Model_ParispIndep
        case "parisp_indep_detach": model = Model_ParispIndepDetach
        case "parisp_naive":        model = Model_ParispNaive
        case "parisp_naive_indep":  model = Model_ParispNaiveIndep
        case "parisp_hyper":        model = Model_ParispHyper
        case "parisp_hyper_indep":  model = Model_ParispHyperIndep
    model = model(**vars(args))

    if args.pretrained is not None:
        model.raw2rgb.load_state_dict(torch.load(args.pretrained[0]))
        model.rgb2raw.load_state_dict(torch.load(args.pretrained[1]))

    print(f"Logging to {args.logs_dir}")
    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.postfix_logs, version=args.mode)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir, monitor="val/psnr/rgb", mode="max", auto_insert_metric_name=False,
            save_top_k=args.ckpt_topk, save_last=True,
            filename="psnr={val/psnr/rgb:.4f}_{epoch:03d}"),
        ModelCheckpoint(
            dirpath=args.ckpt_dir, save_top_k=-1, every_n_epochs=args.ckpt_freq, auto_insert_metric_name=False,
            filename="{epoch:03d}_psnr={val/psnr/rgb:.4f}"),
    ]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, profiler=args.profiler)

    if args.mode == "train":
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_ckpt)

        trainer.save_checkpoint(f"{args.ckpt_dir}/final.ckpt")
        Path(args.wght_dir).mkdir(parents=True, exist_ok=True)
        model.export_weights(args.wght_dir)
    elif args.mode == "test":
        results = trainer.test(model, datamodule=data, ckpt_path=args.resume_ckpt)
        utils.savejson(results, f"{args.logs_dir}/results.json")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


class _Data(pl.LightningDataModule):
    def prepare_data(self):
        self.train_dataset(check=True)
        self.val_dataset(check=True)
        self.test_dataset(check=True)

    def train_dataset(self, check=False): ...
    def val_dataset(self, check=False): ...
    def test_dataset(self, check=False): ...

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset(),
            batch_size=self.hparams.batch_size, shuffle=True, drop_last=True,
            num_workers=self.hparams.data_workers, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset(),
            batch_size=self.hparams.batch_size, shuffle=False, drop_last=True,
            num_workers=self.hparams.data_workers, pin_memory=False, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset(),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.hparams.data_workers)


class Data_RealBlurRaw(_Data):
    def __init__(self, batch_size, data_workers, data_cache, crop_size,
                 data_rbr,   list_rbr_train,   list_rbr_val,   list_rbr_test,
                 **kwds):
        super().__init__()
        self.save_hyperparameters()

    def train_dataset(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_train,
            cache_dir=self.hparams.data_cache + "/realblur_raw",
            crop_config=datasets.raw.cached.RandomCropConfig(self.hparams.crop_size + 5),
            unify_bayer=True, skip_check=not check)

    def val_dataset(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_val,
            cache_dir=self.hparams.data_cache + "/realblur_raw",
            crop_config=datasets.raw.cached.CenterCropConfig(self.hparams.crop_size + 5),
            unify_bayer=True, skip_check=not check)

    def test_dataset(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.data_rbr, self.hparams.list_rbr_test,
            cache_dir=self.hparams.data_cache + "/realblur_raw",
            unify_bayer=True, skip_check=not check)


class Data_Raise(_Data):
    def __init__(self, batch_size, data_workers, data_cache, crop_size,
                 data_raise, list_raise_train, list_raise_val, list_raise_test,
                 **kwds):
        super().__init__()
        self.save_hyperparameters()

    def train_dataset(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_train,
            cache_dir=self.hparams.data_cache + "/raise_raw",
            crop_config=datasets.raw.cached.RandomCropConfig(self.hparams.crop_size + 5),
            unify_bayer=True, skip_check=not check)

    def val_dataset(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_test,
            cache_dir=self.hparams.data_cache + "/raise_raw",
            crop_config=datasets.raw.cached.CenterCropConfig(self.hparams.crop_size + 5),
            unify_bayer=True, skip_check=not check)

    def test_dataset(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.data_raise, self.hparams.list_raise_test,
            cache_dir=self.hparams.data_cache + "/raise_raw",
            unify_bayer=True, skip_check=not check)


class Data_All(_Data):
    def __init__(self, batch_size, data_workers, data_cache, crop_size,
                 data_rbr,   list_rbr_train,   list_rbr_val,   list_rbr_test,
                 data_raise, list_raise_train, list_raise_val, list_raise_test,
                 **kwds):
        super().__init__()
        self.save_hyperparameters()

        self.data_rbr = Data_RealBlurRaw(
            batch_size=batch_size, data_workers=data_workers, data_cache=data_cache, crop_size=crop_size,
            data_rbr=data_rbr, list_rbr_train=list_rbr_train,
            list_rbr_val=list_rbr_val, list_rbr_test=list_rbr_test,
            **kwds)
        self.data_raise = Data_Raise(
            batch_size=batch_size, data_workers=data_workers, data_cache=data_cache, crop_size=crop_size,
            data_raise=data_raise, list_raise_train=list_raise_train,
            list_raise_val=list_raise_val, list_raise_test=list_raise_test,
            **kwds)

    def train_dataset(self, check=False):
        rbr_train   = self.data_rbr.train_dataset(check=check)
        raise_train = self.data_raise.train_dataset(check=check)
        return datasets.raw.Concatenate(rbr_train, raise_train)

    def val_dataset(self, check=False):
        rbr_val   = self.data_rbr.val_dataset(check=check)
        raise_val = self.data_raise.val_dataset(check=check)
        return datasets.raw.Concatenate(rbr_val, raise_val)

    def test_dataset(self, check=False):
        rbr_test   = self.data_rbr.test_dataset(check=check)
        raise_test = self.data_raise.test_dataset(check=check)
        return datasets.raw.Concatenate(rbr_test, raise_test)


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


class Model_ParispIndep(pl.LightningModule):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.rgb2raw = models.paramisp.BasicPipeline(3, 3, 24)
        self.raw2rgb = models.paramisp.BasicPipeline(3, 3, 24)

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

        if self.hparams.save_image:
            Path(f"{self.hparams.logs_dir}/raw").mkdir(parents=True, exist_ok=True)
            Path(f"{self.hparams.logs_dir}/rgb").mkdir(parents=True, exist_ok=True)

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

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix)
        loss_rgb = self.compute_loss(rgb_est, rgb)

        optim_raw2rgb.zero_grad()
        self.manual_backward(loss_rgb)
        optim_raw2rgb.step()

        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix)
        loss_raw = self.compute_loss(raw_est, raw)

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
            utils.imsave(plot, f"{self.hparams.logs_dir}/val_latest.png")

    def test_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)

        H, W = raw.shape[-2:]
        H = H // 2 * 2
        W = W // 2 * 2
        raw = raw[:, :, :H, :W]
        rgb = rgb[:, :, :H, :W]
        bayer_mask = bayer_mask[:, :, :H, :W]

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix)
        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix)

        met_rgb = self.compute_metrics(rgb_est, rgb)
        met_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "test/loss/rgb":  met_rgb["loss"],
            "test/psnr/rgb":  met_rgb["psnr"],
            "test/ssim/rgb":  met_rgb["ssim"],
            "test/mssim/rgb": met_rgb["mssim"],
            "test/loss/raw":  met_raw["loss"],
            "test/psnr/raw":  met_raw["psnr"],
            "test/ssim/raw":  met_raw["ssim"],
            "test/mssim/raw": met_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if self.hparams.save_image:
            for i in range(len(names)):
                name = f"{camera_ids[i]}_{names[i]}".replace("/", "_")
                utils.savetiff(raw_est[i], f"{self.hparams.logs_dir}/raw/{name}.tiff")
                utils.imsave(rgb_est[i],   f"{self.hparams.logs_dir}/rgb/{name}.png")

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

    def export_weights(self, save_dir: str, postfix: str = None):
        if postfix is not None:
            postfix = f"_{postfix}"
        torch.save(self.raw2rgb.state_dict(), f"{save_dir}/raw2rgb{postfix}.pt")
        torch.save(self.rgb2raw.state_dict(), f"{save_dir}/rgb2raw{postfix}.pt")


class Model_ParispNaive(Model_ParispIndep):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds):
        super().__init__(lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.camera_keys = [
            "SONY/ILCE-7RM3",
            "NIKON CORPORATION/NIKON D7000",
            "NIKON CORPORATION/NIKON D90",
            "NIKON CORPORATION/NIKON D40",
        ]

        self.rgb2raw_model = nn.ModuleDict({
            k: models.paramisp.BasicPipeline(3, 3, 24, n_dab=1)
            for k in self.camera_keys})
        self.raw2rgb_model = nn.ModuleDict({
            k: models.paramisp.BasicPipeline(3, 3, 24, n_dab=1)
            for k in self.camera_keys})

    def process_raw2rgb(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        y = x
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.demosaic(y, bayer_mask)
        y = self.colormatrix(y, color_matrix)

        # if self.current_epoch > 10:
        y_ = [None] * len(camera_ids)
        for i, camera_id in enumerate(camera_ids):
            if camera_id in self.camera_keys:
                y_[i] = self.raw2rgb_model[camera_id](y[i].unsqueeze(0))
        y = torch.cat(y_, dim=0)

        y = self.raw2rgb(y)
        y = self.linear2srgb(y)
        return y

    def process_rgb2raw(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        y = x
        y = self.srgb2linear(y)
        y = self.rgb2raw(y)

        y_ = [None] * len(camera_ids)
        for i, camera_id in enumerate(camera_ids):
            if camera_id in self.camera_keys:
                y_[i] = self.rgb2raw_model[camera_id](y[i].unsqueeze(0))
        y = torch.cat(y_, dim=0)

        y = self.colormatrix(y, torch.inverse(color_matrix))
        y = self.mosaic(y, bayer_mask)
        y = self.whitebalance(y, 1 / white_balance, bayer_mask)
        return y

    def training_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)
        optim_raw2rgb, optim_rgb2raw = self.optimizers()

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix, camera_ids)
        loss_rgb = self.compute_loss(rgb_est, rgb)

        optim_raw2rgb.zero_grad()
        self.manual_backward(loss_rgb)
        optim_raw2rgb.step()

        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix, camera_ids)
        loss_raw = self.compute_loss(raw_est, raw)

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

    def validation_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix, camera_ids)
        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix, camera_ids)

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
            utils.imsave(plot, f"{self.hparams.logs_dir}/val_latest.png")

    def test_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)

        rgb_est = self.process_raw2rgb(raw, bayer_mask, white_balance, color_matrix, camera_ids)
        raw_est = self.process_rgb2raw(rgb, bayer_mask, white_balance, color_matrix, camera_ids)

        met_rgb = self.compute_metrics(rgb_est, rgb)
        met_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "test/loss/rgb":  met_rgb["loss"],
            "test/psnr/rgb":  met_rgb["psnr"],
            "test/ssim/rgb":  met_rgb["ssim"],
            "test/mssim/rgb": met_rgb["mssim"],
            "test/loss/raw":  met_raw["loss"],
            "test/psnr/raw":  met_raw["psnr"],
            "test/ssim/raw":  met_raw["ssim"],
            "test/mssim/raw": met_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if self.hparams.save_image:
            for i in range(len(names)):
                name = f"{camera_ids[i]}_{names[i]}".replace("/", "_")
                utils.savetiff(raw_est[i], f"{self.hparams.logs_dir}/raw/{name}.tiff")
                utils.imsave(rgb_est[i],   f"{self.hparams.logs_dir}/rgb/{name}.png")

    def configure_optimizers(self):
        models_raw2rgb = [self.raw2rgb, *list(self.raw2rgb_model.values())]
        models_rgb2raw = [self.rgb2raw, *list(self.rgb2raw_model.values())]
        params_raw2rgb = itertools.chain(*[model.parameters() for model in models_raw2rgb])
        params_rgb2raw = itertools.chain(*[model.parameters() for model in models_rgb2raw])
        optim_raw2rgb = torch.optim.Adam(params_raw2rgb, lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        optim_rgb2raw = torch.optim.Adam(params_rgb2raw, lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        sched_raw2rgb = torch.optim.lr_scheduler.StepLR(
            optim_raw2rgb, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        sched_rgb2raw = torch.optim.lr_scheduler.StepLR(
            optim_rgb2raw, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        return [optim_raw2rgb, optim_rgb2raw], [sched_raw2rgb, sched_rgb2raw]

    def export_weights(self, save_dir: str, postfix: str = None):
        if postfix is not None:
            postfix = f"_{postfix}"
        torch.save(self.raw2rgb.state_dict(), f"{save_dir}/raw2rgb{postfix}.pt")
        torch.save(self.rgb2raw.state_dict(), f"{save_dir}/rgb2raw{postfix}.pt")


class Model_ParispNaiveIndep(Model_ParispNaive):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds):
        super().__init__(lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.rgb2raw_model = models.paramisp.BasicPipeline(3, 3, 24, n_dab=1)
        self.raw2rgb_model = models.paramisp.BasicPipeline(3, 3, 24, n_dab=1)

    def process_raw2rgb(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        y = x
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.demosaic(y, bayer_mask)
        y = self.colormatrix(y, color_matrix)

        y = self.raw2rgb_model(y)

        y = self.raw2rgb(y)
        y = self.linear2srgb(y)
        return y

    def process_rgb2raw(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        y = x
        y = self.srgb2linear(y)
        y = self.rgb2raw(y)

        y = self.rgb2raw_model(y)

        y = self.colormatrix(y, torch.inverse(color_matrix))
        y = self.mosaic(y, bayer_mask)
        y = self.whitebalance(y, 1 / white_balance, bayer_mask)
        return y

    def configure_optimizers(self):
        models_raw2rgb = [self.raw2rgb, self.raw2rgb_model]
        models_rgb2raw = [self.rgb2raw, self.rgb2raw_model]
        params_raw2rgb = itertools.chain(*[model.parameters() for model in models_raw2rgb])
        params_rgb2raw = itertools.chain(*[model.parameters() for model in models_rgb2raw])
        optim_raw2rgb = torch.optim.Adam(params_raw2rgb, lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        optim_rgb2raw = torch.optim.Adam(params_rgb2raw, lr=self.hparams.lr,
                                         weight_decay=self.hparams.weight_decay)
        sched_raw2rgb = torch.optim.lr_scheduler.StepLR(
            optim_raw2rgb, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        sched_rgb2raw = torch.optim.lr_scheduler.StepLR(
            optim_rgb2raw, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        return [optim_raw2rgb, optim_rgb2raw], [sched_raw2rgb, sched_rgb2raw]


class Model_ParispHyper(Model_ParispNaive):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds):
        super().__init__(lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, save_image, **kwds)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.embedding = nn.Embedding(16, 4)
        self.rgb2raw_hyper = models.paramisp.HyperPipeline(4, 3, 3, 24, n_dab=1)
        self.raw2rgb_hyper = models.paramisp.HyperPipeline(4, 3, 3, 24, n_dab=1)

    def camera_keys(self, camera_id):
        camera_keys = {
            "SONY/ILCE-7RM3": 1,
            "NIKON CORPORATION/NIKON D7000": 2,
            "NIKON CORPORATION/NIKON D90": 3,
            "NIKON CORPORATION/NIKON D40": 4,
        }
        if camera_id in camera_keys:
            return camera_keys[camera_id]
        return 0

    def process_raw2rgb(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        camera_keys = [self.camera_keys(camera_id) for camera_id in camera_ids]
        print(camera_keys)
        y = x
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.demosaic(y, bayer_mask)
        y = self.colormatrix(y, color_matrix)
        y = self.raw2rgb_hyper(y, self.embedding(camera_keys))
        y = self.raw2rgb(y)
        y = self.linear2srgb(y)
        return y

    def process_rgb2raw(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        camera_keys = [self.camera_keys(camera_id) for camera_id in camera_ids]
        print(camera_keys)
        y = x
        y = self.srgb2linear(y)
        y = self.rgb2raw(y)
        y = self.rgb2raw_hyper(y, self.embedding(camera_keys))
        y = self.colormatrix(y, torch.inverse(color_matrix))
        y = self.mosaic(y, bayer_mask)
        y = self.whitebalance(y, 1 / white_balance, bayer_mask)
        return y


class Model_ParispHyperIndep(Model_ParispHyper):
    def process_raw2rgb(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        camera_ids = ["" for camera_id in camera_ids]
        return super().process_raw2rgb(x, bayer_mask, white_balance, color_matrix, camera_ids)

    def process_rgb2raw(self, x: torch.Tensor, bayer_mask: torch.Tensor,
                        white_balance: torch.Tensor, color_matrix: torch.Tensor, camera_ids) -> torch.Tensor:
        camera_ids = ["" for camera_id in camera_ids]
        return super().process_rgb2raw(x, bayer_mask, white_balance, color_matrix, camera_ids)


class Model_ParispIndepDetach(Model_ParispIndep):
    def training_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, color_matrix, camera_ids, names = self.destruct_batch(batch)
        optim_raw2rgb, optim_rgb2raw = self.optimizers()

        raw_mid = self.process_raw2mid(raw, bayer_mask, white_balance, color_matrix)
        rgb_mid = self.process_rgb2mid(rgb, bayer_mask, white_balance, color_matrix)

        rgb_mid_est = self.raw2rgb(raw_mid)
        loss_rgb = self.compute_loss(rgb_mid_est, rgb_mid)

        optim_raw2rgb.zero_grad()
        self.manual_backward(loss_rgb)
        optim_raw2rgb.step()

        raw_mid_est = self.rgb2raw(rgb_mid)
        loss_raw = self.compute_loss(raw_mid_est, raw_mid)

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


class Model_CyispWB(pl.LightningModule):
    def __init__(self, lr, lr_step, lr_gamma, weight_decay, momentum, logs_dir, **kwds):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.rgb2raw = models.paramisp.BasicPipeline(3, 1, 24)
        self.raw2rgb = models.paramisp.GuidedPipeline(1, 3, 24)
        self.clrattn = models.paramisp.ColorFeatureExtractor(3, 24)

        self.whitebalance = utils.color.WhiteBalance()

    def destruct_batch(self, batch: dict):
        raw: torch.Tensor = batch["raw"]  # B, 1, H, W
        rgb: torch.Tensor = batch["rgb"]  # B, 3, H, W
        bayer_mask:    torch.Tensor = batch["bayer_mask"].unsqueeze(1).to(torch.int64)    # B, 1, H, W
        white_balance: torch.Tensor = batch["white_balance"].unsqueeze(-1).unsqueeze(-1)  # B, 3, 1, 1
        camera_ids: list[str] = batch["camera_id"]  # B
        names: list[str] = batch["name"]  # B

        raw = raw[..., 2:-2, 2:-2]
        rgb = rgb[..., 2:-2, 2:-2]
        bayer_mask = bayer_mask[..., 2:-2, 2:-2]

        return raw, rgb, bayer_mask, white_balance, camera_ids, names

    def process_rgb2raw(self, raw: torch.Tensor, rgb: torch.Tensor,
                        bayer_mask: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
        y = rgb
        y = self.rgb2raw(y)
        y = self.whitebalance(y, 1 / white_balance, bayer_mask)
        return y

    def process_raw2rgb(self, raw: torch.Tensor, rgb: torch.Tensor,
                        bayer_mask: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
        t = self.clrattn(rgb)

        y = raw
        y = self.whitebalance(y, white_balance, bayer_mask)
        y = self.raw2rgb(y, t)
        return y

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_con = F.l1_loss(x, y)
        loss_fft = F.l1_loss(torch.fft.fft2(x, norm="backward"),
                             torch.fft.fft2(y, norm="backward"))
        return loss_con + 0.1 * loss_fft

    def training_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, camera_ids, names = self.destruct_batch(batch)
        optim_raw2rgb, optim_rgb2raw = self.optimizers()

        rgb_est = self.process_raw2rgb(raw, rgb, bayer_mask, white_balance)
        loss_rgb = self.compute_loss(rgb_est, rgb)

        optim_raw2rgb.zero_grad()
        self.manual_backward(loss_rgb)
        optim_raw2rgb.step()

        raw_est = self.process_rgb2raw(raw, rgb, bayer_mask, white_balance)
        loss_raw = self.compute_loss(raw_est, raw)

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

    def training_epoch_end(self, outputs):
        sched_raw2rgb, sched_rgb2raw = self.lr_schedulers()
        sched_raw2rgb.step()
        sched_rgb2raw.step()

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
        raw, rgb, bayer_mask, white_balance, camera_ids, names = self.destruct_batch(batch)

        rgb_est = self.process_raw2rgb(raw, rgb, bayer_mask, white_balance)
        raw_est = self.process_rgb2raw(raw, rgb, bayer_mask, white_balance)

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
            utils.imsave(plot, f"{self.hparams.logs_dir}/val_latest.png")

    def test_step(self, batch: dict, batch_idx: int):
        raw, rgb, bayer_mask, white_balance, camera_ids, names = self.destruct_batch(batch)

        H, W = raw.shape[-2:]
        H = H // 2 * 2
        W = W // 2 * 2
        raw = raw[:, :, :H, :W]
        rgb = rgb[:, :, :H, :W]
        bayer_mask = bayer_mask[:, :, :H, :W]

        rgb_est = self.process_raw2rgb(raw, rgb, bayer_mask, white_balance)
        raw_est = self.process_rgb2raw(raw, rgb, bayer_mask, white_balance)

        met_rgb = self.compute_metrics(rgb_est, rgb)
        met_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "test/loss/rgb":  met_rgb["loss"],
            "test/psnr/rgb":  met_rgb["psnr"],
            "test/ssim/rgb":  met_rgb["ssim"],
            "test/mssim/rgb": met_rgb["mssim"],
            "test/loss/raw":  met_raw["loss"],
            "test/psnr/raw":  met_raw["psnr"],
            "test/ssim/raw":  met_raw["ssim"],
            "test/mssim/raw": met_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if self.hparams.save_image:
            for i in range(len(names)):
                name = f"{camera_ids[i]}_{names[i]}".replace("/", "_")
                utils.savetiff(raw_est[i], f"{self.hparams.logs_dir}/raw/{name}.tiff")
                utils.imsave(rgb_est[i],   f"{self.hparams.logs_dir}/rgb/{name}.png")

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

    def export_weights(self, save_dir: str, postfix: str = None):
        if postfix is not None:
            postfix = f"_{postfix}"
        torch.save(self.rgb2raw.state_dict(), f"{save_dir}/rgb2raw{postfix}.pt")
        torch.save(self.raw2rgb.state_dict(), f"{save_dir}/raw2rgb{postfix}.pt")
        torch.save(self.clrattn.state_dict(), f"{save_dir}/clrattn{postfix}.pt")


class Model_Cyisp(Model_CyispWB):
    def process_rgb2raw(self, raw: torch.Tensor, rgb: torch.Tensor,
                        bayer_mask: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
        y = rgb
        y = self.rgb2raw(y)
        return y

    def process_raw2rgb(self, raw: torch.Tensor, rgb: torch.Tensor,
                        bayer_mask: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
        t = self.clrattn(rgb)

        y = raw
        y = self.raw2rgb(y, t)
        return y


if __name__ == "__main__":
    main()
