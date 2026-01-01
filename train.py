import sys

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F

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
parser.add_argument('--dataset_path', required=True, type=str, help='Path to dataset')
args = parser.parse_args()

# ---- Load config ----
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

distributed_training_enabled = config.get('distributed_training_enabled', False)
world_size = config.get('world_size', 1)
logs_dir = config.get('tensorboard_logs_dir', '/runs')
image_size = config.get('image_size', 256)
batch_size = config.get('batch_size', 4)
num_workers = config.get('num_workers', 4)

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    print("LOCAL_RANK not found, defaulting to 0")
    local_rank = 0

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
image_tansform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

label_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.PILToTensor(), 
])

# ---- Dataset ----
dataset = TripletDataset(args.dataset_path, 'train', image_transform=image_tansform, label_transform=label_transform, apply_augmentation=False)

if distributed_training_enabled and world_size > 1:
    dataloader = prepare(dataset, torch.distributed.get_rank(), world_size, batch_size, num_workers=num_workers)
else:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ---- Models ----
interp = Interpolator(sem_c=7, base_c=64).to(device)
seg = Segmentator().to(device)

if distributed_training_enabled and world_size > 1:
    interp = nn.parallel.DistributedDataParallel(
        interp, device_ids=[local_rank]
    )
    seg = nn.parallel.DistributedDataParallel(
        seg, device_ids=[local_rank]
    )

# ---- Losses and optimizers ----

ce_loss = nn.CrossEntropyLoss()
losses = {
    "ce": lambda preds, targets: ce_loss(preds, targets),
    "dice": lambda preds, targets: loss.multiclass_dice_loss(preds, targets),
}

optimizers = {
    "seg": optim.Adam(seg.parameters(), lr=1e-4, betas=(0.5, 0.999)),
    "interp": optim.Adam(interp.parameters(), lr=1e-4, betas=(0.5, 0.999)),
}

weights = {
    'seg' : {
        'ce': 1.0,
        'dice': 1.0,
    },
    'interp' : {
        'ce': 1.0,
        'dice': 1.0,
    }
}

print("Starting training...")

def prepare_label(label, device, num_classes=7):
    """
    label: Tensor of shape [B, 1, H, W] or [B, H, W]
    returns: one-hot tensor [B, C, H, W]
    """
    label = label.to(device)
    label = F.one_hot(
        label.squeeze(1).long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    return label

def prepare_interp_input(label, image, device, num_classes=7):
    """
    label: Tensor of shape [B, 1, H, W] or [B, H, W]
    returns: one-hot tensor [B, C, H, W]
    """
    label = prepare_label(label, device, num_classes)
    image = image.to(device)
    return torch.cat([image, label], dim=1).to(device)

def train(losses, optimizers, dataloader, device, writer):
    epochs = 50
    for epoch in range(epochs):
        for data in dataloader:
            images = data['images'] 
            labels = data['labels'] 
            for i in range(3):
                seg_img = images[i].to(device)
                seg_label = prepare_label(labels[i], device)
                optimizers['seg'].zero_grad()
                seg_output = seg(seg_img)
                seg_loss_ce = losses['ce'](seg_output, seg_label) * weights['seg']['ce']
                seg_loss_dice = losses['dice'](seg_output, seg_label) * weights['seg']['dice']
                seg_loss = seg_loss_ce + seg_loss_dice
                seg_loss.backward()
                optimizers['seg'].step()

            optimizers['interp'].zero_grad()

            input_frame_a = prepare_interp_input(labels[0], images[0], device)
            input_frame_b = prepare_interp_input(labels[2], images[2], device)
            target_mid_mask = prepare_label(labels[1], device)
            generated_mid = interp(input_frame_a, input_frame_b)
            interp_loss_ce = losses['ce'](generated_mid, target_mid_mask.argmax(dim=1)) * weights['interp']['ce']
            interp_loss_dice = losses['dice'](generated_mid, target_mid_mask) * weights['interp']['dice']
            interp_loss = interp_loss_ce + interp_loss_dice
            interp_loss.backward()
            optimizers['interp'].step()



if __name__ == "__main__":
    if distributed_training_enabled and world_size > 1:
        setup(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
        writer = SummaryWriter(log_dir=logs_dir) if torch.distributed.get_rank() == 0 else None
        fn = lambda: train(losses, optimizers, dataloader, device, writer)
        run_parallel(fn, world_size)
        cleanup()
    else:
        train(losses, optimizers, dataloader, device, writer)

