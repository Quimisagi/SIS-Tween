from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from .dataset_base import DatasetBase


class TripletDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        image_transform=None,
        label_transform=None,
        apply_augmentation=False,
        max_samples=None,
    ):
        self.loader = DatasetBase(root_dir, mode)
        self.image_transform = image_transform or transforms.ToTensor()
        self.label_transform = label_transform or transforms.ToTensor()
        self.apply_augmentation = apply_augmentation

        # expose class info
        self.classes = self.loader.classes
        self.class_to_idx = self.loader.class_to_idx

        # --------------------------------------------
        # Build SAME-CLASS CONSECUTIVE triplets
        # --------------------------------------------
        self.triplet_indices = []
        self.triplet_class = []

        for class_id, idxs in self.loader.class_to_indices.items():
            if len(idxs) < 3:
                continue

            # keep original order for consecutive triplets
            for i in range(len(idxs) - 2):
                self.triplet_indices.append(
                    (idxs[i], idxs[i + 1], idxs[i + 2])
                )
                self.triplet_class.append(class_id)

        # optionally limit number of triplets (DEV mode)
        if max_samples is not None and len(self.triplet_indices) > max_samples:
            self.triplet_indices = self.triplet_indices[:max_samples]
            self.triplet_class = self.triplet_class[:max_samples]

    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, index):
        i0, i1, i2 = self.triplet_indices[index]
        samples = [self.loader[i0], self.loader[i1], self.loader[i2]]

        images = [s["image"] for s in samples]
        labels = [s["label"] for s in samples]
        edges  = [s["edge"]  for s in samples]

        cls = self.triplet_class[index]

        if self.apply_augmentation:
            images, labels, edges = self.augment_triplet(images, labels, edges)

        if self.image_transform:
            images = [self.image_transform(img) for img in images]
        if self.label_transform:
            labels = [self.label_transform(lbl) for lbl in labels]
            edges  = [self.label_transform(edg) for edg in edges]

        return {
            "basename": [s["basename"] for s in samples],
            "images": images,
            "labels": labels,
            "edges": edges,
            "class_index": torch.tensor(cls, dtype=torch.long),
        }

    def augment_triplet(self, images, labels, edges):
        do_hflip = torch.rand(1).item() < 0.35
        angle = torch.empty(1).uniform_(-10, 10).item()

        out_images, out_labels, out_edges = [], [], []
        for img, lbl, edg in zip(images, labels, edges):
            if do_hflip:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
                edg = TF.hflip(edg)

            img = TF.rotate(img, angle)
            lbl = TF.rotate(lbl, angle)
            edg = TF.rotate(edg, angle)

            out_images.append(img)
            out_labels.append(lbl)
            out_edges.append(edg)

        return out_images, out_labels, out_edges
