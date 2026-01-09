import os
import argparse
import yaml
import torch.nn as nn
from pathlib import Path
from diffusers import AutoencoderKL

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization, TrainConfig
import engine.setup as setup
from engine import losses, Trainer, RuntimeContext, DataloaderBundle
from models import Interpolator, Segmentator, Synthesizer
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

    vae = AutoencoderKL.from_single_file(cfg.autoencoder_path).to(device)
    synthesizer = Synthesizer(vae).to(device)

    if cfg.distributed_enabled and world_size > 1:
        interp = nn.parallel.DistributedDataParallel(interp, device_ids=[local_rank])
        seg = nn.parallel.DistributedDataParallel(seg, device_ids=[local_rank])

    # ---- Loss / Optimizers ----
    seg_loss = losses.CompositeLoss({
        "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
        "ce":   (losses.CrossEntropyLoss().to(device), 1.0),
    }).to(device)
    interp_loss = losses.CompositeLoss({
        "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
        "ce":   (losses.CrossEntropyLoss().to(device), 1.0),
    }).to(device)
    synth_loss = losses.CompositeLoss({
        "perceptual": (losses.PerceptualLoss().to(device), 1.0),
    }).to(device)

    loss = losses.MultitaskLoss(
        seg=seg_loss,
        interp=interp_loss,
        synth=synth_loss,
    )


    # ---- TensorBoard (rank 0 only) ----
    writer = visualization.init_tensorboard(cfg, local_rank)

    # ---- Train ----
    logger.info("Starting training...")
    context = RuntimeContext(
        device=device,
        writer=writer,
        logger=logger,
    )
    dataloaders = DataloaderBundle(
            train=dataloader_train,
            val=dataloader_val,
        )
    trainer = Trainer(
        loss_fn=loss,
        dataloaders=dataloaders,
        cfg=cfg,
        context=context,
    )

    trainer.train()

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
