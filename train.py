import os
import yaml
import argparse
import torch.nn as nn
from pathlib import Path

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization, TrainConfig
import engine.setup as setup
from engine import train_loop, Loss, RuntimeContext, DataloaderBundle, TrainingState
from models import Interpolator, Segmentator
from logs.logger import init_logger

import torch.distributed as dist

def is_dist():
    return (
        "RANK" in os.environ and
        "WORLD_SIZE" in os.environ
    )

def train_fn(cfg, args):

    if cfg.distributed_enabled and is_dist():
        dist.init_process_group(backend="nccl")

    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    device, local_rank = setup.prepare_device(cfg.distributed_enabled, world_size)

    logger = init_logger("train", f"training_rank{local_rank}.log")
    logger.debug(f"Rank {local_rank}/{world_size} using device {device}")

    logger.info("Seting up training...")
    logger.debug(f"Configuration: {cfg}")

    # ---- Transforms ----
    image_transform, label_transform = setup.create_transforms(cfg.image_size)

    # ---- Dataset / Loader ----
    logger.info("Preparing Dataset")
    dataset_train = TripletDataset(
        args.dataset_path,
        "train",
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False,
    )

    dataloader_train = distributed_gpu.prepare(
        dataset_train,
        local_rank,
        world_size,
        cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    logger.info("Training Dataset and DataLoader ready")
    logger.debug(f"Training Dataset size: {len(dataset_train)}")

    dataset_val = TripletDataset(
        args.dataset_path,
        "val",
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False,
    )
    dataloader_val = distributed_gpu.prepare(
        dataset_val,
        local_rank,
        world_size,
        cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    logger.info("Validation Dataset and DataLoader ready")
    logger.debug(f"Validation Dataset size: {len(dataset_val)}")

    # ---- Models ----
    interp = Interpolator(sem_c=6, base_c=64).to(device)
    seg = Segmentator().to(device)

    if cfg.distributed_enabled and world_size > 1:
        interp = nn.parallel.DistributedDataParallel(interp, device_ids=[local_rank])
        seg = nn.parallel.DistributedDataParallel(seg, device_ids=[local_rank])

    # ---- Loss / Optimizers ----
    loss = Loss()

    optimizers, schedulers = setup.create_optimizers(
        {"seg": seg, "interp": interp},
        lr_seg=1e-4,
        lr_interp=1e-4,
    )

    weights = setup.create_weights()

    # ---- TensorBoard (rank 0 only) ----
    writer = visualization.init_tensorboard(cfg, local_rank)

    # ---- Train ----
    logger.info("Starting training...")
    context_bundle = RuntimeContext(
        device=device,
        writer=writer,
        logger=logger,
    )
    training_state = TrainingState(
        loss=loss,
        weights=weights,
        optimizers=optimizers,
        schedulers=schedulers,
        seg=seg,
        interp=interp,
    )
    dataloader_bundle = DataloaderBundle(
        train=dataloader_train,
        val=dataloader_val,
    )
    train_loop(
        training_state,
        context_bundle,
        dataloader_bundle,
        cfg,
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
