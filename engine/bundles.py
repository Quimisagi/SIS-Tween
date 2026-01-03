from dataclasses import dataclass
from typing import Any, List
import torch
import torch.nn as nn


@dataclass
class ContextBundle:
    device: torch.device
    writer: Any
    logger: Any
    epochs: int
    segmentator_score_threshold: float

@dataclass
class Batch:
    images: List[torch.Tensor]
    labels: List[torch.Tensor]

@dataclass
class ModelsBundle:
    seg: nn.Module
    interp: nn.Module

@dataclass
class DataloaderBundle:
    train: Any
    val: Any
