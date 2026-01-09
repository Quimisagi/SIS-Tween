from dataclasses import dataclass
from typing import Any, List
import torch
import torch.nn as nn


@dataclass
class RuntimeContext:
    device: torch.device
    writer: Any
    logger: Any


@dataclass
class TrainingState:
    loss: Any
    optimizers: dict
    schedulers: dict
    seg: nn.Module
    interp: nn.Module
    synth: nn.Module


@dataclass
class Batch:
    images: List[torch.Tensor]
    labels: List[torch.Tensor]


@dataclass
class DataloaderBundle:
    train: Any
    val: Any
