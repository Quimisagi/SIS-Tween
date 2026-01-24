import torch.distributed as dist
from pathlib import Path
import torch

from data.triplet_dataset import TripletDataset
from utils import distributed_gpu, visualization
import engine.setup as setup
from engine import losses, Trainer, RuntimeContext, DataloaderBundle
from logs.logger import init_logger
from config import read_arguments

def test_fn(opt):
    # ---- Distributed ----
    if opt.distributed_enabled and distributed_gpu.is_dist():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device, local_rank = setup.prepare_device(opt.distributed_enabled, world_size)

    logger = init_logger("test", f"test_rank{local_rank}.log")
    logger.debug(f"Rank {local_rank}/{world_size} using device {device}")

    logger.info("Setting up testing...")

    # ---- Transforms ----
    image_transform, label_transform = setup.create_transforms(opt.image_size)

    logger.info("Preparing Test Dataset")

    dataset_test = TripletDataset(
        opt.dataset_path,
        "val", 
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False, # Important: No augmentation for testing
    )

    dataloader_test = distributed_gpu.prepare(
        dataset_test,
        local_rank,
        world_size,
        opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False
    )

    logger.info(f"Test Dataset size: {len(dataset_test)}")

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
                "ce": (losses.CrossEntropyLoss().to(device), 1.0),
                "sobel": (losses.SobelEdgeLoss().to(device), 0.5),
                "entropy": (losses.EntropyLoss().to(device), 0.05),
            }
        ).to(device)

    loss = losses.MultitaskLoss(**tasks)

    # ---- TensorBoard ----
    writer = visualization.init_tensorboard(opt, local_rank) if local_rank == 0 else None

    # ---- Runtime Context ----
    context = RuntimeContext(
        device=device,
        writer=writer,
        logger=logger,
        world_size=world_size,
        local_rank=local_rank,
    )

    # ---- Dataloader Bundle ----
    dataloaders = DataloaderBundle(
        train=None, 
        val=dataloader_test,
    )

    trainer = Trainer(
        loss_fn=loss,
        dataloaders=dataloaders,
        opt=opt,  
        context=context,
    )

    # ---- Execution ----
    trainer.test()

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    opt = read_arguments(train=False)

    assert Path(opt.dataset_path).exists(), (
        f"Dataset path {opt.dataset_path} does not exist."
    )

    test_fn(opt)


if __name__ == "__main__":
    main()
