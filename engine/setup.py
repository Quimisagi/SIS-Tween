import os
import torch
import torchvision.transforms as transforms

def prepare_device(distributed, world_size):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed and world_size > 1:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0

def create_transforms(image_size):
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.PILToTensor(),
    ])

    return image_transform, label_transform

def create_optimizers(models, lr_seg=1e-4, lr_interp=1e-4):
    optimizers = {
        'seg': torch.optim.Adam(models['seg'].parameters(), lr=lr_seg),
        'interp': torch.optim.Adam(models['interp'].parameters(), lr=lr_interp),
    }
    return optimizers

def create_weights():
    weights = {
        "seg": {"ce": 1.0, "dice": 1.0},
        "interp": {"ce": 1.0, "dice": 1.0},
    }
    return weights
