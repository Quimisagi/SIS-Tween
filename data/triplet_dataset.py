import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .dataset_base import DatasetBase


class TripletDataset(Dataset):
    def __init__(self, root_dir, mode="train", image_transform=None, label_transform=None, apply_augmentation=False):
        self.loader = DatasetBase(root_dir, mode)
        self.image_transform = image_transform or transforms.ToTensor()
        self.label_transform = label_transform or transforms.ToTensor()
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.loader) - 2

    def __getitem__(self, index):
        samples = [
            self.loader[index],
            self.loader[index + 1],
            self.loader[index + 2],
        ]

        images = [s["image"] for s in samples]
        labels = [s["label"] for s in samples]

        if self.apply_augmentation:
            images, labels = self.augment_triplet(images, labels)

        if self.image_transform:
            images = [self.image_transform(img) for img in images]
        if self.label_transform:
            labels = [self.label_transform(lbl) for lbl in labels]

        return {
            "basename": [s["basename"] for s in samples],
            "images": images, 
            "labels": labels,
        }

    def augment_triplet(self, images, labels):
        do_hflip = random.random() < 0.35
        angle = random.uniform(-10, 10)

        out_images = []
        out_labels = []

        for img, lbl in zip(images, labels):
            if do_hflip:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)

            img = TF.rotate(img, angle)
            lbl = TF.rotate(lbl, angle)

            out_images.append(img)
            out_labels.append(lbl)

        return out_images, out_labels
