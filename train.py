import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.triplet_dataset import TripletDataset
import argparse

from utils.distributed_gpu import prepare, run_parallel, setup, cleanup
from utils.train_config import TrainConfig
import engine.setup as setup
from engine import train_loop, Loss
from models import Interpolator, Segmentator

def main():
    # ---- Argument parser ----
    parser = argparse.ArgumentParser(description='Train GANiMate')
    parser.add_argument('--config', type=str, default='configuration.yaml')
    parser.add_argument('--dataset_path', required=True)
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = TrainConfig(**config)

    # ---- Device / distributed ----
    device, local_rank = setup.prepare_device(cfg.distributed_enabled, cfg.world_size)

    # ---- Logging ----
    writer = SummaryWriter(log_dir=cfg.tensorboard_logs_dir) if (
        not cfg.distributed_enabled or local_rank == 0
    ) else None

    # ---- Transforms ----
    image_transform, label_transform = setup.create_transforms(cfg.image_size)

    # ---- Dataset / Loader ----
    dataset = TripletDataset(
        args.dataset_path,
        'train',
        image_transform=image_transform,
        label_transform=label_transform,
        apply_augmentation=False
    )

    if cfg.distributed_enabled and cfg.world_size > 1:
        dataloader = prepare(
            dataset,
            local_rank,
            cfg.world_size,
            cfg.batch_size,
            num_workers=cfg.num_workers
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )

    # ---- Models ----
    interp = Interpolator(sem_c=7, base_c=64).to(device)
    seg = Segmentator().to(device)

    if cfg.distributed_enabled and cfg.world_size > 1:
        interp = nn.parallel.DistributedDataParallel(interp, device_ids=[local_rank])
        seg = nn.parallel.DistributedDataParallel(seg, device_ids=[local_rank])

    # ---- Loss / Optimizers / Weights ----
    loss = Loss()

    optimizers = setup.create_optimizers(
        {'seg': seg, 'interp': interp},
        lr_seg=1e-4,
        lr_interp=1e-4
    )

    weights = setup.create_weights()

    # ---- Train ----
    print("Starting training...")
    train_loop(seg, interp, loss, optimizers, dataloader, device, writer, weights)

if __name__ == "__main__":
    main()

