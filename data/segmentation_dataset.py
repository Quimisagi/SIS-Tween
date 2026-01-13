import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch

from .dataset_base import DatasetBase


class SegmentationDataset(Dataset):
    def __init__(
        self, root_dir, mode="train", transform=None, apply_augmentation=False
    ):
        self.loader = DatasetBase(root_dir, mode)
        self.transform = transform or transforms.ToTensor()
        self.apply_augmentation = apply_augmentation

        self.classes = self.loader.classes
        self.class_to_idx = self.loader.class_to_idx

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        sample = self.loader[index]

        image = sample["image"]
        label = sample["label"]
        edge  = sample["edge"]
        class_id = sample["class"]

        if self.apply_augmentation:
            image, label, edge = self.augment(image, label, edge)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            edge  = self.transform(edge)

        return {
            "basename": sample["basename"],
            "image": image,
            "label": label,
            "edge": edge,
            "class": torch.tensor(class_id, dtype=torch.long),
        }

    def augment(self, image, label, edge):
        do_hflip = random.random() < 0.35
        angle = random.uniform(-10, 10)

        if do_hflip:
            image = TF.hflip(image)
            label = TF.hflip(label)
            edge  = TF.hflip(edge)

        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)
        edge  = TF.rotate(edge, angle)

        return image, label, edge
