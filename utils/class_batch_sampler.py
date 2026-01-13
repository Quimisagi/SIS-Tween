import math
import random
import torch
from torch.utils.data import Sampler
from collections import defaultdict


class DistributedClassBatchSampler(Sampler):
    """
    Distributed batch sampler that guarantees:
      - ONE class per batch
      - Correct behavior with TripletDataset (triplet-level indexing)
      - DDP-safe sharding
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=True,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # --------------------------------------------------
        # IMPORTANT FIX:
        # dataset indices are TRIPLET indices, NOT loader indices
        # --------------------------------------------------
        class_to_indices = defaultdict(list)
        for ds_idx, (i0, _, _) in enumerate(dataset.triplet_indices):
            cls = dataset.loader[i0]["class"]
            class_to_indices[cls].append(ds_idx)

        self.class_to_indices = class_to_indices
        self.classes = list(class_to_indices.keys())

        self.batches = self._create_batches()

        # batches per replica
        self.num_batches = math.ceil(len(self.batches) / self.num_replicas)

    def _create_batches(self):
        batches = []

        classes = self.classes.copy()
        if self.shuffle:
            random.shuffle(classes)

        for cls in classes:
            indices = self.class_to_indices[cls]
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        random.seed(epoch)
        self.batches = self._create_batches()

    def __iter__(self):
        # shard batches across GPUs
        return iter(self.batches[self.rank :: self.num_replicas])

    def __len__(self):
        return self.num_batches
