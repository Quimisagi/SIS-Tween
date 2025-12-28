import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .dataset_base import DatasetBase


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None, apply_augmentation=True):
        self.loader = DatasetBase(root_dir, mode)
        self.transform = transform or transforms.ToTensor()
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        sample = self.loader[index]

        image = sample["image"]
        label = sample["label"]

        if self.apply_augmentation:
            image, label = self.augment(image, label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return {
            "basename": sample["basename"],
            "image": image,
            "label": label,
        }

    def augment(self, image, label):
        do_hflip = random.random() < 0.35
        angle = random.uniform(-10, 10)

        if do_hflip:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)

        return image, label
