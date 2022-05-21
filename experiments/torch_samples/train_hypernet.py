#!/usr/bin/env python
import argparse
from pathlib import Path
import random
from warnings import warn
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms
import torchvision.datasets
import torchvision.models
import torchmetrics
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
from models.paramisp import HyperConv


parser = argparse.ArgumentParser(description="HyperNet Sample",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("save_name", nargs="?", default="EMNIST_letter_classifier", type=str, metavar="NAME",
                    help="name to save results")
parser.add_argument("--max-epochs", "--epochs", default=90, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--min-epochs", metavar="N", type=int, default=1,
                    help="force to run at least this many epochs")
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W")
parser.add_argument("--scheduler-step", metavar="N", type=int, default=30,
                    help="step size for scheduler")
parser.add_argument("--scheduler-gamma", metavar="G", type=float, default=0.1,
                    help="gamma for scheduler")
parser.add_argument("-j", "--data-workers", default=4, type=int, metavar="N",
                    help="number of data loading workers per GPU")
parser.add_argument("--seed", default=None, type=int,
                    help="random seed for initializing training")
parser.add_argument("--resume", nargs="?", const="auto", default=None, type=str, metavar="PATH",
                    help="path to import latest checkpoint (default: none)")
parser.add_argument("--pretrained", default=None, type=str, metavar="PATH",
                    help="path to import model weights only (default: None)")
parser.add_argument("--save-root", default=f"{utils.path.root.as_posix()}/results", type=str, metavar="PATH",
                    help="parent directory to save results")
parser.add_argument("--save-freq", default=5, type=int, metavar="N",
                    help="save model every N epochs")
parser.add_argument("--profiler", choices=["simple", "advanced", "pytorch"], default="simple",
                    help="enable torch profiler")


def parse_args():
    args = parser.parse_args()

    args.save_dir = Path(args.save_root, args.save_name).as_posix()
    args.ckpt_dir = f"{args.save_dir}/checkpoints"
    args.wght_dir = f"{args.save_dir}/weights"
    Path(args.wght_dir).mkdir(parents=True, exist_ok=True)

    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume == "auto":
        args.resume = f"{args.ckpt_dir}/last.ckpt"
    args.resume     = utils.path.purify(args.resume)
    args.pretrained = utils.path.purify(args.pretrained)

    args.gpus = torch.cuda.device_count()
    args.deterministic = args.seed is not None

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        pl.seed_everything(args.seed)
        warn("You have chosen to seed training. "
             "This will turn on the CUDNN deterministic setting, "
             "which can slow down your training considerably! "
             "You may see unexpected behavior when restarting "
             "from checkpoints.")

    data = Data(**vars(args))
    model = Model(**vars(args))

    if args.pretrained:
        model.model.load_state_dict(torch.load(args.pretrained))

    print(f"Logging to {args.save_dir}/train/{args.timestamp}")
    logger = TensorBoardLogger(save_dir=args.save_dir, name="train", version=args.timestamp)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            monitor="val/accuracy",
            mode="max",
            save_top_k=3,
            filename="acc={val/accuracy:.4f}__" + args.timestamp + "__{epoch:03d}",
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            save_top_k=-1,
            every_n_epochs=args.save_freq,
            filename=(args.timestamp + "__{epoch:03d}__acc={val/accuracy:.4f}"),
            auto_insert_metric_name=False,
        ),
    ]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, profiler=args.profiler)

    trainer.fit(model, datamodule=data, ckpt_path=args.resume)

    trainer.save_checkpoint(f"{args.ckpt_dir}/final__{args.timestamp}.ckpt")
    torch.save(model.model.state_dict(), f"{args.wght_dir}/classifier__{args.timestamp}.pt")


class Model(pl.LightningModule):
    def __init__(self, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 1e-4,
                 sched_step: int = 30, sched_gamma: float = 0.1, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = ClassifierNet()
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, labels = batch
        labels -= 1

        output: torch.Tensor = self(inputs)
        loss: torch.Tensor = self.loss_fn(output, labels)

        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({
            "step": self.current_epoch,
            "train/loss/epoch": loss,
        }, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, labels = batch
        labels -= 1

        output: torch.Tensor = self(inputs)
        loss: torch.Tensor = self.loss_fn(output, labels)
        acc: torch.Tensor = self.acc_fn(output, labels)

        self.log_dict({
            "step": self.current_epoch,
            "val/loss": loss,
            "val/accuracy": acc,
        }, on_step=False, on_epoch=True)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.lr,
            momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.scheduler_step, gamma=self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]


class Data(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256, workers: int = 4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        self.trainset = torchvision.datasets.EMNIST(
            split="letters", root="data/downloads", download=True, train=True, transform=self.transform)
        self.testset = torchvision.datasets.EMNIST(
            split="letters", root="data/downloads", download=True, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size, shuffle=True,  # prefer False for DistributedSampler
            num_workers=self.hparams.data_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.data_workers, pin_memory=True,
        )


class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = HyperConv(in_channels=1, out_channels=10, kernel_size=3, h_length=16)
        self.pool1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = HyperConv(in_channels=10, out_channels=100, kernel_size=3, h_length=16)
        self.pool2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=100 * 7 * 7, out_features=1000),
            nn.Linear(in_features=1000, out_features=26),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor = torch.ones(10)) -> torch.Tensor:
        x = self.conv1(x, h)
        x = self.pool1(x)
        x = self.conv2(x, h)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.layers2(x)
        return x


if __name__ == "__main__":
    main()
