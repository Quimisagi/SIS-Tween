import argparse
import yaml
from pathlib import Path
import torch.distributed as dist

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization, TrainConfig
import engine.setup as setup
from engine import losses, Trainer, RuntimeContext, DataloaderBundle
from logs.logger import init_logger


def train_fn(cfg, args):

    if cfg.distributed_enabled and distributed_gpu.is_dist():
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

    # ---- Loss / Optimizers ----
    tasks = {}

    if "seg" in cfg.active_models:
        tasks["seg"] = losses.CompositeLoss(
            {
                "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
                "ce": (losses.CrossEntropyLoss().to(device), 1.0),
            }
        ).to(device)

    if "interp" in cfg.active_models:
        tasks["interp"] = losses.CompositeLoss(
            {
                "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
                "ce": (losses.CrossEntropyLoss().to(device), 1.0),
            }
        ).to(device)

    if "synth" in cfg.active_models:
        tasks["synth"] = losses.CompositeLoss(
            {
                "perceptual": (losses.PerceptualLoss().to(device), 1.0),
            }
        ).to(device)

    loss = losses.MultitaskLoss(**tasks)

    # ---- TensorBoard (rank 0 only) ----
    writer = visualization.init_tensorboard(cfg, local_rank)

    # ---- Train ----
    logger.info("Starting training...")
    context = RuntimeContext(
        device=device,
        writer=writer,
        logger=logger,
        world_size=world_size,
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

    assert Path(
        args.dataset_path
    ).exists(), f"Dataset path {args.dataset_path} does not exist."
    assert Path(args.config).exists(), f"Config file {args.config} does not exist."

    with open(args.config) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    train_fn(cfg, args)


if __name__ == "__main__":
    main()
