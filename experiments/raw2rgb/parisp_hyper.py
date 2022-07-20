#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import os
import multiprocessing as mp
from datetime import datetime
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchmetrics.functional import \
    peak_signal_noise_ratio as compute_psnr, \
    structural_similarity_index_measure as compute_ssim, \
    multiscale_structural_similarity_index_measure as compute_mssim
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
parser.add_argument("mode", choices=["train-content", "train-hyper", "test"], help="train or test")
parser.add_argument("data", choices=["lg2", "sm2", "sm1", "dslr4"], nargs="?", help="Dataset to train hypernet")
parser.add_argument("--disable-id", action="store_true", help="Disable camera id inputs")
parser.add_argument("-o", "--save-name", metavar="NAME", default="RAW2RGB_HyperParISP",     help="name to save results")              # noqa: E501
parser.add_argument("-O", "--save-root", metavar="PATH", default=utils.path.get("results"), help="parent directory to save results")  # noqa: E501
parser.add_argument("--cache-root",     metavar="PATH", default=utils.env.get("DATA_CACHE"),   help="path to cache the splitted dataset")  # noqa: E501
parser.add_argument("--dataroot-rbr",   metavar="PATH", default=utils.env.get("REALBLUR_RAW"), help="path to the RealBlur RAW dataset")  # noqa: E501
parser.add_argument("--dataroot-raise", metavar="PATH", default=utils.env.get("RAISE"),        help="path to the RAISE dataset")         # noqa: E501
parser.add_argument("--datalist-rbr-train", metavar="PATH", default=utils.path.get("data/datalist/realblur_raw_all.train.csv"))  # noqa: E501
parser.add_argument("--datalist-rbr-val",   metavar="PATH", default=utils.path.get("data/datalist/realblur_raw_all.val.csv"))    # noqa: E501
parser.add_argument("--datalist-rbr-test",  metavar="PATH", default=utils.path.get("data/datalist/realblur_raw_all.test.csv"))   # noqa: E501
parser.add_argument("--datalist-raise-d7000-train", metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d7000.train.csv"))  # noqa: E501
parser.add_argument("--datalist-raise-d7000-val",   metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d7000.val.csv"))    # noqa: E501
parser.add_argument("--datalist-raise-d7000-test",  metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d7000.test.csv"))   # noqa: E501
parser.add_argument("--datalist-raise-d90-train", metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d90.train.csv"))  # noqa: E501
parser.add_argument("--datalist-raise-d90-val",   metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d90.val.csv"))    # noqa: E501
parser.add_argument("--datalist-raise-d90-test",  metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d90.test.csv"))   # noqa: E501
parser.add_argument("--datalist-raise-d40-train", metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d40.train.csv"))  # noqa: E501
parser.add_argument("--datalist-raise-d40-val",   metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d40.val.csv"))    # noqa: E501
parser.add_argument("--datalist-raise-d40-test",  metavar="PATH", default=utils.path.get("data/datalist/raise_raw_d40.test.csv"))   # noqa: E501
parser.add_argument("--crop-size",              metavar="N", default=512, type=int, help="crop size of the input image")            # noqa: E501
parser.add_argument("-j", "--data-workers",     metavar="N", default=4,   type=int, help="number of data loading workers per GPU")  # noqa: E501
parser.add_argument("--max-epochs", "--epochs", metavar="N", default=300, type=int, help="number of total epochs to run")  # noqa: E501
parser.add_argument("-b", "--batch-size",       metavar="N", default=4,   type=int, help="batch size")                     # noqa: E501
parser.add_argument("--lr",           metavar="LR", default=1e-4, type=float, help="learning rate")
parser.add_argument("--lr-step",      nargs="+",    default=20,   type=int,   help="scheduler step")
parser.add_argument("--lr-gamma",     metavar="G",  default=0.9,  type=float, help="scheduler gamma")
parser.add_argument("--momentum",     metavar="M",  default=0.9,  type=float, help="momentum")
parser.add_argument("--weight-decay", metavar="W",  default=1e-8, type=float, help="weight decay")
parser.add_argument("--seed", default=None, type=int, help="random seed for initializing training")
parser.add_argument("--resume", metavar="NAME", nargs="?", const="last.ckpt",  default=None, type=str, help="checkpoint to resume if exists")            # noqa: E501
parser.add_argument("--load",   metavar="NAME", nargs="?", const="final.ckpt", default=None, type=str, help="checkpoint to load weight only if exists")  # noqa: E501
parser.add_argument("--ckpt-epoch", metavar="N", default=50, type=int, help="save checkpoint every N epochs")                # noqa: E501
parser.add_argument("--ckpt-best",  metavar="N", default=3,  type=int, help="save checkpoints for the best N performances")  # noqa: E501
parser.add_argument("--profiler", choices=["simple", "advanced", "pytorch"], default=None, help="profiler to use")  # noqa: E501
parser.add_argument("--save-image", action="store_true", default=False, help="save image when validating or testing")  # noqa: E501


def main():
    args = parser.parse_args()

    if not hasattr(args, "timestamp"):
        args.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.data_workers < 0:
        args.data_workers = mp.cpu_count() // (2 * torch.cuda.device_count())

    args.gpus = torch.cuda.device_count()
    args.deterministic = args.seed is not None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        pl.seed_everything(args.seed)
        warn("You have chosen to seed training. "
             "This will turn on the CUDNN deterministic setting, which can slow down your training considerably! "
             "You may see unexpected behavior when restarting from checkpoints.")

    args.save_dir = Path(args.save_root, args.save_name).as_posix()
    args.ckpt_postfix = "checkpoints"
    args.logs_postfix = "logs"
    args.ckpt_dir = f"{args.save_dir}/{args.ckpt_postfix}"
    args.logs_dir = f"{args.save_dir}/{args.logs_postfix}/{args.mode}"

    if args.resume is not None:
        args.resume = utils.path.purify(args.resume)
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"{args.resume=} does not exist")
    if args.load is not None:
        args.load = utils.path.purify(args.load)
        if not Path(args.load).exists():
            raise FileNotFoundError(f"{args.load=} does not exist")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "train-content":
        args.mode = "train"
        args.train_mode = "content"
    elif args.mode == "train-hyper":
        args.mode = "train"
        args.train_mode = "hyper"
    else:
        args.train_mode = None

    print(args)

    match args.mode:
        case "train": train(args)
        case "test": test(args)
        case _: raise ValueError(f"unknown mode: {args.mode}")


def train(args):
    model = Model(args)

    if args.data is not None:
        match args.data:
            case "lg2":   data = Data_TrainLarge(args)
            case "sm2":   data = Data_TrainSmall(args)
            case "sm1":   data = Data_TrainMini(args)
            case "dslr4": data = Data_TrainDSLRAll(args)
            case _: raise ValueError(f"unknown data: {args.data}")
    else:
        data = Data_TrainLarge(args)

    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.logs_postfix,
                               version=f"{args.mode}-{args.train_mode}")

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir, monitor="val/rgb/psnr", mode="max", auto_insert_metric_name=False,
            save_top_k=args.ckpt_best, save_last=True,
            filename=args.train_mode + "-{val/rgb/psnr:.2f}-{epoch:03d}"),
        ModelCheckpoint(
            dirpath=args.ckpt_dir, save_top_k=-1, every_n_epochs=args.ckpt_epoch, auto_insert_metric_name=False,
            filename=args.train_mode + "-{epoch:03d}"),
    ]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, profiler=args.profiler)

    results = trainer.fit(model, datamodule=data, ckpt_path=args.resume)
    utils.saveyaml(results, f"{args.logs_dir}/results.yaml")
    trainer.save_checkpoint(f"{args.ckpt_dir}/{args.train_mode}-final.ckpt")


def test(args):
    model = Model(args)

    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.logs_postfix, version=args.mode)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, profiler=args.profiler)

    results = trainer.test(model, datamodule=Data_TestA7R3(args))
    utils.saveyaml(results, f"{args.logs_dir}/results_a7r3.yaml")
    results = trainer.test(model, datamodule=Data_TestD7000(args))
    utils.saveyaml(results, f"{args.logs_dir}/results_d7000.yaml")
    results = trainer.test(model, datamodule=Data_TestD90(args))
    utils.saveyaml(results, f"{args.logs_dir}/results_d90.yaml")
    results = trainer.test(model, datamodule=Data_TestD40(args))
    utils.saveyaml(results, f"{args.logs_dir}/results_d40.yaml")


class Data_Base(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)

        self.root_rbr   = self.hparams.cache_root + "/realblur_raw"
        self.root_raise = self.hparams.cache_root + "/raise_raw"

    def prepare_data(self):
        self.train_dataset(check=True)
        self.val_dataset(check=True)
        self.test_dataset(check=True)

    def train_dataset(self, check=False): ...
    def val_dataset(self, check=False): ...
    def test_dataset(self, check=False): ...

    def train_dataset_args(self, check=False):
        return {
            "skip_check": not check,
            "crop_config": datasets.raw.cached.RandomCropConfig(self.hparams.crop_size + 5),
            "unify_bayer": True,
        }

    def val_dataset_args(self, check=False):
        return {
            "skip_check": not check,
            "crop_config": datasets.raw.cached.CenterCropConfig(self.hparams.crop_size + 5),
            "unify_bayer": True,
        }

    def test_dataset_args(self, check=False):
        return {
            "skip_check": not check,
            "unify_bayer": True,
        }

    def train_dataset_a7r3(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.dataroot_rbr, self.hparams.datalist_rbr_train,
            cache_dir=self.root_rbr, **self.train_dataset_args(check=check))

    def train_dataset_d7000(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d7000_train,
            cache_dir=self.root_raise, **self.train_dataset_args(check=check))

    def train_dataset_d90(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d90_train,
            cache_dir=self.root_raise, **self.train_dataset_args(check=check))

    def train_dataset_d40(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d40_train,
            cache_dir=self.root_raise, **self.train_dataset_args(check=check))

    def val_dataset_a7r3(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.dataroot_rbr, self.hparams.datalist_rbr_val,
            cache_dir=self.root_rbr, **self.val_dataset_args(check=check))

    def val_dataset_d7000(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d7000_val,
            cache_dir=self.root_raise, **self.val_dataset_args(check=check))

    def val_dataset_d90(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d90_val,
            cache_dir=self.root_raise, **self.val_dataset_args(check=check))

    def val_dataset_d40(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d40_val,
            cache_dir=self.root_raise, **self.val_dataset_args(check=check))

    def test_dataset_a7r3(self, check=False):
        return datasets.raw.cached.RealBlurRaw(
            self.hparams.dataroot_rbr, self.hparams.datalist_rbr_test,
            cache_dir=self.root_rbr, **self.test_dataset_args(check=check))

    def test_dataset_d7000(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d7000_test,
            cache_dir=self.root_raise, **self.test_dataset_args(check=check))

    def test_dataset_d90(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d90_test,
            cache_dir=self.root_raise, **self.test_dataset_args(check=check))

    def test_dataset_d40(self, check=False):
        return datasets.raw.cached.RAISERaw(
            self.hparams.dataroot_raise, self.hparams.datalist_raise_d40_test,
            cache_dir=self.root_raise, **self.test_dataset_args(check=check))

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


class Data_TrainLarge(Data_Base):
    def train_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.train_dataset_a7r3(check=check),
            self.train_dataset_d7000(check=check),
        )

    def val_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.val_dataset_a7r3(check=check),
            self.val_dataset_d7000(check=check),
        )


class Data_TrainSmall(Data_Base):
    def train_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.train_dataset_d90(check=check),
            self.train_dataset_d40(check=check),
        )

    def val_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.val_dataset_d90(check=check),
            self.val_dataset_d40(check=check),
        )


class Data_TrainMini(Data_Base):
    def train_dataset(self, check=False):
        return self.train_dataset_d40(check=check)

    def val_dataset(self, check=False):
        return self.val_dataset_d40(check=check)


class Data_TrainDSLRAll(Data_Base):
    def train_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.train_dataset_a7r3(check=check),
            self.train_dataset_d7000(check=check),
            self.train_dataset_d90(check=check),
            self.train_dataset_d40(check=check),
        )

    def val_dataset(self, check=False):
        return datasets.raw.Concatenate(
            self.val_dataset_a7r3(check=check),
            self.val_dataset_d7000(check=check),
            self.val_dataset_d90(check=check),
            self.val_dataset_d40(check=check),
        )


class Data_TestA7R3(Data_Base):
    def test_dataset(self, check=False):
        return self.test_dataset_a7r3(check=check)


class Data_TestD7000(Data_Base):
    def test_dataset(self, check=False):
        return self.test_dataset_d7000(check=check)


class Data_TestD90(Data_Base):
    def test_dataset(self, check=False):
        return self.test_dataset_d90(check=check)


class Data_TestD40(Data_Base):
    def test_dataset(self, check=False):
        return self.test_dataset_d40(check=check)


class Model(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)
        self.automatic_optimization = False

        camera_max_devices = 16
        camera_embedding_dim = 4
        colormatrix_embedding_dim = 9
        whitebalance_embedding_dim = 3
        h_dim = camera_embedding_dim + colormatrix_embedding_dim + whitebalance_embedding_dim

        self.camera_embedding = nn.Embedding(camera_max_devices, camera_embedding_dim)
        self.forward_net = models.paramisp.Hyper2StagePipeline(h_dim, 3, 3, 24)
        self.inverse_net = models.paramisp.Hyper2StagePipeline(h_dim, 3, 3, 24)

        match (self.hparams.mode, self.hparams.train_mode):
            case ("train", "content"):
                self.hypertrain(False)
            case ("train", "hyper"):
                self.hypertrain(True)
                self.load_model(remove_hyper=True)
            case ("test", _):
                self.hypertrain(True)
                self.load_model()
            case _:
                raise ValueError(f"Invalid mode: {self.hparams.mode}/{self.hparams.train_mode}")

        self.whitebalance = utils.color.WhiteBalance()
        self.demosaic     = utils.bayer.Demosaic5x5()
        self.mosaic       = utils.bayer.Mosaic()
        self.colormatrix  = utils.color.ColorMatrix()
        self.gamma        = utils.color.Linear2SRGB()
        self.linearize    = utils.color.SRGB2Linear()

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

        self.register_buffer("D65_TO_D50", D65_TO_D50)
        self.register_buffer("XYZ_TO_SRGB_D65", XYZ_TO_SRGB_D65)

        if self.hparams.mode == "train":
            Path(f"{self.hparams.logs_dir}/images/val").mkdir(parents=True, exist_ok=True)
        elif self.hparams.mode == "test":
            Path(f"{self.hparams.logs_dir}/images/raw").mkdir(parents=True, exist_ok=True)
            Path(f"{self.hparams.logs_dir}/images/rgb").mkdir(parents=True, exist_ok=True)

    def hypertrain(self, enable: bool):
        self.hypertrain_enabled = enable
        self.forward_net.hypertrain(enable)
        self.inverse_net.hypertrain(enable)

    def load_model(self, remove_hyper: bool = False):
        states = utils.loadpt(self.hparams.load)["state_dict"]
        del states["D65_TO_D50"]
        del states["XYZ_TO_SRGB_D65"]
        if remove_hyper:
            for key in [*states.keys()]:
                if ".affine_weights." in key:
                    del states[key]

        self.load_state_dict(states, strict=not remove_hyper)

    def configure_optimizers(self):
        optim_fwd = torch.optim.Adam(
            self.forward_net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optim_inv = torch.optim.Adam(
            self.inverse_net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_fwd = torch.optim.lr_scheduler.StepLR(
            optim_fwd, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        sched_inv = torch.optim.lr_scheduler.StepLR(
            optim_inv, self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        return [optim_fwd, optim_inv], [sched_fwd, sched_inv]

    def camera_id(self, camera_name: str):
        camera_name = camera_name.upper().split("/")[-1]
        camera_names = {
            "ILCE-7RM3": 1,
            "NIKON D7000": 2,
            "NIKON D90": 3,
            "NIKON D40": 4,
        }
        return camera_names.get(camera_name, 0)

    def destruct_batch(self, batch: dict):
        raw: torch.Tensor = batch["raw"]
        rgb: torch.Tensor = batch["rgb"]
        mask: torch.Tensor = batch["bayer_mask"].unsqueeze(1).to(torch.int64)    # B, 1, H, w
        wb:   torch.Tensor = batch["white_balance"].unsqueeze(-1).unsqueeze(-1)  # B, 3, 1, 1

        cmat: torch.Tensor = self.XYZ_TO_SRGB_D65 @ self.D65_TO_D50 @ batch["color_matrix"]  # B, 3, 3
        cmat = cmat / torch.mean(cmat, dim=-1, keepdim=True)

        camera_ids = torch.tensor([(self.camera_id(camera_name) if not self.hparams.disable_id else 0)
                                   for camera_name in batch["camera_id"]],
                                  device=raw.device)

        image_names: list[str] = batch["name"]

        return raw, rgb, mask, wb, cmat, camera_ids, image_names

    def forward_process(self, raw: torch.Tensor, mask: torch.Tensor,
                        wb: torch.Tensor, cmat: torch.Tensor, camera_ids: torch.Tensor):
        rgb = raw
        rgb = self.whitebalance(rgb, wb, mask)
        rgb = self.demosaic(rgb, mask)
        rgb = self.colormatrix(rgb, cmat)

        if self.hypertrain_enabled:
            h_id   = self.camera_embedding(camera_ids).flatten(start_dim=1)
            h_cmat = cmat.flatten(start_dim=1)
            h_wb   = wb.flatten(start_dim=1)
            h = torch.cat([h_id, h_cmat, h_wb], dim=1)
        else:
            h = None
        rgb = self.forward_net(rgb, h)

        rgb = self.gamma(rgb)
        return rgb

    def inverse_process(self, rgb: torch.Tensor, mask: torch.Tensor,
                        wb: torch.Tensor, cmat: torch.Tensor, camera_ids: torch.Tensor):
        raw = rgb
        raw = self.linearize(raw)

        if self.hypertrain_enabled:
            h_id   = self.camera_embedding(camera_ids).flatten(start_dim=1)
            h_cmat = cmat.flatten(start_dim=1)
            h_wb   = wb.flatten(start_dim=1)
            h = torch.cat([h_id, h_cmat, h_wb], dim=1)
        else:
            h = None
        raw = self.inverse_net(raw, h)

        raw = self.colormatrix(raw, cmat.inverse())
        raw = self.mosaic(raw, mask)
        raw = self.whitebalance(raw, 1 / wb, mask)
        return raw

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        return image[..., 2:-2, 2:-2]

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.crop(x)
        y = self.crop(y)
        loss_content = F.l1_loss(x, y)
        loss_fft     = F.l1_loss(torch.fft.fft2(x), torch.fft.fft2(y))
        return loss_content + 0.1 * loss_fft

    def compute_metrics(self, x: torch.Tensor, y: torch.Tensor):
        x = self.crop(x).clip(0, 1)
        y = self.crop(y).clip(0, 1)
        return {
            "loss":  self.compute_loss(x, y),
            "psnr":  compute_psnr(x,  y, data_range=1.),
            "ssim":  compute_ssim(x,  y, data_range=1.),
            "mssim": compute_mssim(x, y, data_range=1.),
        }

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        raw, rgb, mask, wb, cmat, camera_ids, image_names = self.destruct_batch(batch)
        optim_fwd, optim_inv = self.optimizers()

        rgb_est = self.forward_process(raw, mask, wb, cmat, camera_ids)
        loss_rgb = self.compute_loss(rgb_est, rgb)

        optim_fwd.zero_grad()
        self.manual_backward(loss_rgb)
        optim_fwd.step()

        raw_est = self.inverse_process(rgb, mask, wb, cmat, camera_ids)
        loss_raw = self.compute_loss(raw_est, raw)

        optim_inv.zero_grad()
        self.manual_backward(loss_raw)
        optim_inv.step()

        self.log_dict({
            "train/rgb/loss": loss_rgb,
            "train/raw/loss": loss_raw,
        }, prog_bar=True)

        self.log_dict({
            "step": self.current_epoch,
            "train-epoch/rgb/loss": loss_rgb,
            "train-epoch/raw/loss": loss_raw,
        }, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        sched_fwd, sched_inv = self.lr_schedulers()
        sched_fwd.step()
        sched_inv.step()

    def validation_step(self, batch: dict, batch_idx: int):
        raw, rgb, mask, wb, cmat, camera_ids, image_names = self.destruct_batch(batch)

        rgb_est = self.forward_process(raw, mask, wb, cmat, camera_ids)
        raw_est = self.inverse_process(rgb, mask, wb, cmat, camera_ids)

        metrics_rgb = self.compute_metrics(rgb_est, rgb)
        metrics_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "val/rgb/loss":  metrics_rgb["loss"],
            "val/rgb/psnr":  metrics_rgb["psnr"],
            "val/rgb/ssim":  metrics_rgb["ssim"],
            "val/rgb/mssim": metrics_rgb["mssim"],
            "val/raw/loss":  metrics_raw["loss"],
            "val/raw/psnr":  metrics_raw["psnr"],
            "val/raw/ssim":  metrics_raw["ssim"],
            "val/raw/mssim": metrics_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if batch_idx == 0 and (self.current_epoch < 10 or self.current_epoch % 5 == 0):
            plot = utils.imgrid([raw, raw_est, rgb, rgb_est])
            utils.imsave(plot, f"{self.hparams.logs_dir}/images/val/{self.current_epoch}.png")

    def test_step(self, batch: dict, batch_idx: int):
        raw, rgb, mask, wb, cmat, camera_ids, image_names = self.destruct_batch(batch)

        H, W = raw.shape[-2:]
        raw   = raw[..., :2 * (H // 2), :2 * (W // 2)]
        rgb   = rgb[..., :2 * (H // 2), :2 * (W // 2)]
        mask = mask[..., :2 * (H // 2), :2 * (W // 2)]

        rgb_est = self.forward_process(raw, mask, wb, cmat, camera_ids)
        raw_est = self.inverse_process(rgb, mask, wb, cmat, camera_ids)

        metrics_rgb = self.compute_metrics(rgb_est, rgb)
        metrics_raw = self.compute_metrics(raw_est, raw)

        self.log_dict({
            "step": self.current_epoch,
            "test/rgb/loss":  metrics_rgb["loss"],
            "test/rgb/psnr":  metrics_rgb["psnr"],
            "test/rgb/ssim":  metrics_rgb["ssim"],
            "test/rgb/mssim": metrics_rgb["mssim"],
            "test/raw/loss":  metrics_raw["loss"],
            "test/raw/psnr":  metrics_raw["psnr"],
            "test/raw/ssim":  metrics_raw["ssim"],
            "test/raw/mssim": metrics_raw["mssim"],
        }, on_step=False, on_epoch=True)

        if self.hparams.save_image:
            camera_names = batch["camera_id"]
            for i in range(len(image_names)):
                name = f"{camera_names[i]}_{image_names[i]}".replace("/", "_")
                utils.savetiff(raw_est[i],               f"{self.hparams.logs_dir}/images/raw/{name}.tiff")
                utils.imsave(utils.im2uint8(rgb_est[i]), f"{self.hparams.logs_dir}/images/rgb/{name}.png")


if __name__ == "__main__":
    main()
