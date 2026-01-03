import yaml
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization, TrainConfig
import engine.setup as setup
from engine import train_loop, Loss
from models import Interpolator, Segmentator
from logs.logger import init_logger

import torch.distributed as dist

def train_fn(cfg, args):
    # ---- Distributed setup ----
    device, local_rank = setup.prepare_device(cfg.distributed_enabled, cfg.world_size)

    if cfg.distributed_enabled and cfg.world_size > 1:
        dist.init_process_group(backend="nccl")

    logger = init_logger("train", f"training_rank{local_rank}.log")
    logger.debug(f"Rank {local_rank}/{cfg.world_size} using device {device}")

    logger.info("Seting up training...")
    logger.debug(f"Configuration: {cfg}")

    # ---- Transforms ----
    image_transform, label_transform = setup.create_transforms(cfg.image_size)

    # ---- Dataset / Loader ----
    logger.info("Preparing Dataset")
    dataset = TripletDataset(
        args.dataset_path,
        "train",
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False,
    )

    dataloader = distributed_gpu.prepare(
        dataset,
        local_rank,
        cfg.world_size,
        cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    logger.info("Dataset and DataLoader ready")
    logger.debug(f"Dataset size: {len(dataset)}")

    # ---- Models ----
    interp = Interpolator(sem_c=6, base_c=64).to(device)
    seg = Segmentator().to(device)

    if cfg.distributed_enabled and cfg.world_size > 1:
        interp = nn.parallel.DistributedDataParallel(interp, device_ids=[local_rank])
        seg = nn.parallel.DistributedDataParallel(seg, device_ids=[local_rank])

    # ---- Loss / Optimizers ----
    loss = Loss()

    optimizers = setup.create_optimizers(
        {"seg": seg, "interp": interp},
        lr_seg=1e-4,
        lr_interp=1e-4,
    )

    weights = setup.create_weights()

    # ---- TensorBoard (rank 0 only) ----
    writer = visualization.init_tensorboard(cfg, local_rank)

    # ---- Train ----
    logger.info("Starting training...")
    train_loop(
        seg,
        interp,
        loss,
        optimizers,
        dataloader,
        device,
        writer,
        logger,
        weights,
    )

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train SIS_Tween")
    parser.add_argument("--config", type=str, default="configuration.yaml")
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    assert Path(args.dataset_path).exists(), f"Dataset path {args.dataset_path} does not exist."
    assert Path(args.config).exists(), f"Config file {args.config} does not exist."

    with open(args.config) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    train_fn(cfg, args)

if __name__ == "__main__":
    main()
