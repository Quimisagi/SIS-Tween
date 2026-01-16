import torch.distributed as dist
from pathlib import Path

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization
import engine.setup as setup
from engine import losses, Trainer, RuntimeContext, DataloaderBundle
from logs.logger import init_logger

from config import read_arguments
from models import VGG19

import torch


def train_fn(opt):

    # ---- Distributed ----
    if opt.distributed_enabled and distributed_gpu.is_dist():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device, local_rank = setup.prepare_device(opt.distributed_enabled, world_size)

    logger = init_logger("train", f"training_rank{local_rank}.log")
    logger.debug(f"Rank {local_rank}/{world_size} using device {device}")

    logger.info("Setting up training...")

    # ---- Transforms ----
    image_transform, label_transform = setup.create_transforms(opt.image_size)

    # ---- Dataset / Loader ----
    logger.info("Preparing Dataset")

    dataset_train = TripletDataset(
        opt.dataset_path,
        "train",
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False,
    )

    dataloader_train = distributed_gpu.prepare(
        dataset_train,
        local_rank,
        world_size,
        opt.batch_size,
        num_workers=opt.num_workers,
    )

    logger.info("Training Dataset and DataLoader ready")
    logger.debug(f"Training Dataset size: {len(dataset_train)}")

    dataset_val = TripletDataset(
        opt.dataset_path,
        "val",
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False,
    )

    dataloader_val = distributed_gpu.prepare(
        dataset_val,
        local_rank,
        world_size,
        opt.batch_size,
        num_workers=opt.num_workers,
    )

    logger.info("Validation Dataset and DataLoader ready")
    logger.debug(f"Validation Dataset size: {len(dataset_val)}")

    # ---- Losses ----
    tasks = {}

    if "seg" in opt.active_models:
        tasks["seg"] = losses.CompositeLoss(
            {
                "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
                "ce": (losses.CrossEntropyLoss().to(device), 1.0),
            }
        ).to(device)

    if "interp" in opt.active_models:
        tasks["interp"] = losses.CompositeLoss(
            {
                "dice": (losses.MulticlassDiceLoss().to(device), 1.0),
                "ce": (losses.CrossEntropyLoss().to(device), 0.5),
                "entropy": (losses.EntropyLoss().to(device), 0.05),
            }
        ).to(device)

    if "synth" in opt.active_models:
        tasks["synth"] = losses.CompositeLoss(
            {
                "l1": (losses.L1Loss().to(device), 1.0),
                "vgg": (losses.VGGLoss(vgg=VGG19()).to(device), 1.0),
                "gan": (losses.OASISGanLoss(opt, device), 0.1),
            }
        ).to(device)
        tasks["disc"] = losses.CompositeLoss(
            {
                "gan": (losses.OASISGanLoss(opt, device), 1.0),
            }
        ).to(device)
    loss = losses.MultitaskLoss(**tasks)

    # ---- TensorBoard (rank 0 only) ----
    writer = visualization.init_tensorboard(opt, local_rank)

    # ---- Training Runtime ----
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
        opt=opt,  
        context=context,
    )

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    opt = read_arguments(train=True)

    assert Path(opt.dataset_path).exists(), (
        f"Dataset path {opt.dataset_path} does not exist."
    )

    train_fn(opt)


if __name__ == "__main__":
    main()
