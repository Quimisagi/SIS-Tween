import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from logger import logger

logger = logger.setup_logger("debug", "debug.log")

logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
# logger.info(f"CUDA available: {torch.cuda.is_available()}")
# logger.info(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return dataloader

def run_parallel(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
