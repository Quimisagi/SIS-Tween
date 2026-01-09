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
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    label_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.PILToTensor(),
        ]
    )

    return image_transform, label_transform
