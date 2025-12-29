import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.triplet_dataset import TripletDataset
import utils
import lpips
import torchvision.models as models
import argparse

from utils.distributed_gpu import prepare, run_parallel, setup, cleanup
from models import Interpolator, Segmentator
import loss

# ---- Argument parser ----
parser = argparse.ArgumentParser(description='Train GANiMate')
parser.add_argument('--config', type=str, default='configuration.yaml', help='Path to configuration file')
parser.add_argument('--dataset_path', type=str, help='Path to dataset')
args = parser.parse_args()

# ---- Load config ----
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

distributed_training_enabled = config.get('distributed_training_enabled', False)
world_size = config.get('world_size', 1)
logs_dir = config.get('tensorboard_logs_dir', '/runs')
image_size = config.get('image_size', 256)
batch_size = config.get('batch_size', 16)
num_workers = config.get('num_workers', 4)

local_rank = int(os.environ["LOCAL_RANK"])


# ---- Device and logger ----
if distributed_training_enabled and world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not distributed_training_enabled or torch.distributed.get_rank() == 0:
    writer = SummaryWriter(log_dir=logs_dir)
else:
    writer = None

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ---- Dataset ----
dataset = TripletDataset(args.dataset_path, 'train', transform=transform, apply_augmentation=False)

if distributed_training_enabled and world_size > 1:
    dataloader = prepare(dataset, torch.distributed.get_rank(), world_size, batch_size, num_workers=num_workers)
else:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ---- Models ----
interp = Interpolator(frame_c=3, base_c=64).to(device)
seg = Segmentator(frame_c=3, base_c=64).to(device)

if distributed_training_enabled and world_size > 1:
    interp = nn.parallel.DistributedDataParallel(
        interp, device_ids=[local_rank]
    )
    seg = nn.parallel.DistributedDataParallel(
        seg, device_ids=[local_rank]
    )

# ---- Losses and optimizers ----
loss_ce = nn.CrossEntropyLoss()
dice_loss = loss.multiclass_dice_loss()

optimizer_seg = optim.Adam(seg.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_interp = optim.Adam(interp.parameters(), lr=1e-4, betas=(0.5, 0.999))

