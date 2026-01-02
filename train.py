import yaml
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.triplet_dataset import TripletDataset
from utils.distributed_gpu import prepare, run_parallel, setup, cleanup
from utils.train_config import TrainConfig
from utils.visualization import setup_tensorboard 
import engine.setup as setup
from engine import train_loop, Loss
from models import Interpolator, Segmentator
from logs.logger import setup_logger


def main():
    logger = setup_logger('train', 'training.log')
    # ---- Argument parser ----
    parser = argparse.ArgumentParser(description='Train SIS_Tween')
    parser.add_argument('--config', type=str, default='configuration.yaml')
    parser.add_argument('--dataset_path', required=True)
    args = parser.parse_args()

    logger.info("Seting up training...")

    # ---- Load config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = TrainConfig(**config)
    logger.debug(f"Configuration: {cfg}")

    # ---- Device / distributed ----
    device, local_rank = setup.prepare_device(cfg.distributed_enabled, cfg.world_size)
    logger.debug(f"Using device: {device}")

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
    logger.info("Dataset and DataLoader ready")
    logger.debug(f"Dataset size: {len(dataset)}")

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
    interp = Interpolator(sem_c=6, base_c=64).to(device)
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

    # ---- TensorBoard ----
    writer = setup_tensorboard(cfg, local_rank)

    # ---- Train ----
    logger.info("Starting training...")
    def train_fn():
        train_loop(
            seg,
            interp,
            loss,
            optimizers,
            dataloader,
            device,
            writer,
            logger,
            weights
        )

    if cfg.distributed_enabled and cfg.world_size > 1:
        run_parallel(train_fn, cfg.world_size)
    else:
        train_fn()

if __name__ == "__main__":
    main()
