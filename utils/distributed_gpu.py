import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from .class_batch_sampler import DistributedClassBatchSampler

# assumes DistributedClassBatchSampler is already defined/imported


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def is_dist():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def cleanup():
    dist.destroy_process_group()

def prepare(
    dataset,
    rank,
    world_size,
    batch_size=32,
    pin_memory=False,
    num_workers=0,
):
    batch_sampler = DistributedClassBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


def run_parallel(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)
