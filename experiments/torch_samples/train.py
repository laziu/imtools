#!/usr/bin/env python
import argparse
import os
import shutil
from pathlib import Path
import random
from pprint import pprint
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
import torchvision.datasets
import torchvision.models
import numpy as np
from tqdm.auto import tqdm

import utils


parser = argparse.ArgumentParser(description="PyTorch Training Sample",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("save_name", nargs="?", default="EMNIST_letter_classifier", type=str, metavar="NAME",
                    help="name to save results")
parser.add_argument("--epochs", default=90, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W")
parser.add_argument("-j", "--data-workers", default=4, type=int, metavar="N",
                    help="number of data loading workers per GPU")
parser.add_argument("--seed", default=None, type=int,
                    help="random seed for initializing training")
parser.add_argument("--no-validate", action="store_true",
                    help="disable validation during training")
parser.add_argument("--dist-backend", default="nccl", choices=["nccl", "gloo"],
                    help="backend for distributed learning")
parser.add_argument("--dist-store", default="tcp://localhost:23182", type=str, metavar="URL",
                    help="store path for distributed learning")
parser.add_argument("--resume", nargs="?", const="auto", default=None, type=str, metavar="PATH",
                    help="path to import latest checkpoint (default: none)")
parser.add_argument("--pretrained", default=None, type=str, metavar="PATH",
                    help="path to import model weights only (default: None)")
parser.add_argument("--save-root", default=f"{utils.path.root.as_posix()}/results", type=str, metavar="PATH",
                    help="parent directory to save results")
parser.add_argument("--save-freq", default=30, type=int, metavar="N",
                    help="save model every N epochs")
parser.add_argument("--profile", action="store_true",
                    help="enable torch profiler")


class Arguments:
    def __init__(self):
        args = parser.parse_args()
        self.save_name:    str   = args.save_name
        self.save_root:    str   = args.save_root
        self.save_freq:    int   = args.save_freq
        self.epochs:       int   = args.epochs
        self.batch_size:   int   = args.batch_size
        self.lr:           float = args.lr
        self.momentum:     float = args.momentum
        self.weight_decay: float = args.weight_decay
        self.data_workers: int   = args.data_workers
        self.seed:         int   = args.seed
        self.validate:     bool  = not args.no_validate
        self.dist_backend: str   = args.dist_backend
        self.dist_store:   str   = args.dist_store
        self.profile:      bool  = args.profile

        self.save_dir = Path(self.save_root, self.save_name).as_posix()

        if args.resume == "auto":
            args.resume = f"{self.save_dir}/ckpt/latest.pt"
        self.resume     = utils.path.purify(args.resume)
        self.pretrained = utils.path.purify(args.pretrained)

        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_id: int = -1


def main():
    args = Arguments()
    pprint(vars(args))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warn("You have chosen to seed training. "
             "This will turn on the CUDNN deterministic setting, "
             "which can slow down your training considerably! "
             "You may see unexpected behavior when restarting "
             "from checkpoints.")

    args.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.n_gpus == 0:
        raise ValueError("At least one GPU is required")

    mp.spawn(main_per_process, nprocs=args.n_gpus, args=(args,))


def main_per_process(gpu_id: int, args: Arguments):
    args.gpu_id = gpu_id
    device = torch.device("cuda", args.gpu_id)
    print(f"[{args.gpu_id+1}/{args.n_gpus}] spawned (cuda:{args.gpu_id})")

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_store,
                            world_size=args.n_gpus, rank=args.gpu_id)
    torch.cuda.set_device(args.gpu_id)

    print(f"[{args.gpu_id+1}] model init")
    model = ClassifierNet().cuda(args.gpu_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_id])

    criterion = nn.CrossEntropyLoss().cuda(args.gpu_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    epoch = 0
    best_loss = 99999999999.
    best_acc  = 0.

    if args.pretrained:
        print(f"Load weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device)

        model.load_state_dict(state["model"])

    if args.resume:
        print(f"Resume from {args.resume}")
        state = torch.load(args.resume, map_location=device)

        epoch = state["epoch"]
        best_loss = state["best_loss"]
        best_acc  = state["best_acc"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    Path(args.save_dir, "ckpt").mkdir(parents=True, exist_ok=True)

    # Data loading
    print(f"[{args.gpu_id+1}] data loading")
    trainset = torchvision.datasets.EMNIST(
        split="letters", root="data", download=True, train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]))
    valset = torchvision.datasets.EMNIST(
        split="letters", root="data", download=True, train=False,  # use test set instead
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, sampler=train_sampler,
        batch_size=args.batch_size, shuffle=False,  # prefer False for DistributedSampler
        num_workers=args.data_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.data_workers, pin_memory=True,
    )

    print(f"[{args.gpu_id+1}] ready")
    dist.barrier()

    profiler = torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{args.save_dir}/profiler"),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) if args.profile else utils.VoidModule()
    profiler.start()

    summary = SummaryWriter(f"{args.save_dir}/logs")

    if args.validate:
        validate(epoch, val_loader, model, summary, args)

    for epoch in range(epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        epoch_loss = train(epoch, train_loader, model, criterion, optimizer, summary, profiler, args)
        epoch_acc = validate(epoch, val_loader, model, summary, args)
        scheduler.step()

        is_best = epoch_acc > best_acc
        best_loss = min(epoch_loss, best_loss)
        best_acc  = max(epoch_acc, best_acc)

        if gpu_id == 0:
            torch.save({
                "epoch": epoch,
                "best_loss": best_loss,
                "best_acc":  best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"{args.save_dir}/ckpt/latest.pt")

            if is_best:
                shutil.copyfile(f"{args.save_dir}/ckpt/latest.pt", f"{args.save_dir}/ckpt/best.pt")

            if epoch % args.save_freq == 0:
                shutil.copyfile(f"{args.save_dir}/ckpt/latest.pt", f"{args.save_dir}/ckpt/epoch_{epoch}.pt")

    if gpu_id == 0:
        torch.save({
            "epoch": epoch,
            "best_loss": best_loss,
            "best_acc":  best_acc,
            "model": model.state_dict(),
        }, f"{args.save_dir}/ckpt/final.pt")

    profiler.stop()
    dist.destroy_process_group()


def train(
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.parallel.DistributedDataParallel,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    summary: SummaryWriter,
    profiler: torch.profiler.profile,
    args: Arguments,
) -> float:
    device = torch.device("cuda", args.gpu_id)
    model.train()

    epoch_loss = utils.AvgMeter()
    with tqdm(desc=f"Train #{epoch}", total=len(dataloader), ncols=80) as pbar:
        for i, (inputs, labels) in enumerate(dataloader):
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            labels: torch.Tensor = labels.to(device, non_blocking=True) - 1  # 0-based for one-hot vector
            batch_size = inputs.size(0)

            output: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            profiler.step()

            epoch_loss.add(loss.item(), batch_size)
            summary.add_scalar("loss", loss.item(), i + (epoch - 1) * len(dataloader))

            pbar.set_postfix(loss=f"{epoch_loss.avg:.4f}")
            pbar.update()

    summary.add_scalar("loss/epoch", epoch_loss.avg, epoch)
    return epoch_loss.avg


@torch.no_grad()
def validate(
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.parallel.DistributedDataParallel,
    summary: SummaryWriter,
    args: Arguments,
) -> float:
    device = torch.device("cuda", args.gpu_id)
    model.eval()

    epoch_acc = utils.AvgMeter()
    with tqdm(desc=f"Val #{epoch}", total=len(dataloader), ncols=80) as pbar:
        for i, (inputs, labels) in enumerate(dataloader):
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            labels: torch.Tensor = labels.to(device, non_blocking=True) - 1  # 0-based for one-hot vector
            batch_size = inputs.size(0)

            output: torch.Tensor = model(inputs)
            accuracy = output.argmax(dim=1).eq(labels).float().mean()

            epoch_acc.add(accuracy.item(), batch_size)
            summary.add_scalar("accuracy", accuracy.item(), i + (epoch - 1) * len(dataloader))

            pbar.set_postfix(acc=f"{epoch_acc.avg:.4f}")
            pbar.update()

    summary.add_scalar("accuracy/epoch", epoch_acc.avg, epoch)
    return epoch_acc.avg


class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=100 * 7 * 7, out_features=1000),
            nn.Linear(in_features=1000, out_features=26),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers1(x)
        x = x.view(x.size(0), -1)
        x = self.layers2(x)
        return x


if __name__ == "__main__":
    main()
