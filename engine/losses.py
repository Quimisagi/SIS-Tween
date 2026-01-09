import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights

class BaseLoss(nn.Module):
    """
    Contract:
        forward(...) -> (loss: Tensor, metrics: dict[str, Tensor])
    """
    pass

class MulticlassDiceLoss(BaseLoss):
    def __init__(self, eps: float = 1e-6, ignore_bg: bool = False):
        super().__init__()
        self.eps = eps
        self.ignore_bg = ignore_bg

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: (B, C, H, W)
        target: (B, C, H, W) one-hot
        """
        probs = F.softmax(pred, dim=1)

        B, C = probs.shape[:2]
        probs = probs.view(B, C, -1)
        target = target.view(B, C, -1)

        intersection = (probs * target).sum(dim=-1)
        union = probs.sum(dim=-1) + target.sum(dim=-1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        if self.ignore_bg:
            dice = dice[:, 1:]

        dice_mean = dice.mean()
        loss = 1.0 - dice_mean

        return loss, {"dice": dice_mean}


class CrossEntropyLoss(BaseLoss):
    def __init__(self, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:  (B, C, H, W)
        target: (B, H, W)
        """
        loss = F.cross_entropy(
            pred,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )
        return loss, {"cross_entropy": loss}


class FocalLoss(BaseLoss):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(
            pred,
            targets,
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return loss, {"focal": loss}


class PerceptualLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(
            weights=VGG19_Weights.IMAGENET1K_V1
        ).features[:36]
        self.vgg.eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, 3, H, W) in [0,1]
        """
        pred = self.normalize(pred)
        target = self.normalize(target)

        fx = self.vgg(pred)
        fy = self.vgg(target)

        loss = F.l1_loss(fx, fy)
        return loss, {"perceptual": loss}


class CompositeLoss(BaseLoss):
    def __init__(self, losses: dict[str, tuple[BaseLoss, float]]):
        """
        losses:
            name -> (loss_module, weight)
        """
        super().__init__()
        self.losses = nn.ModuleDict({
            name: loss for name, (loss, _) in losses.items()
        })
        self.weights = {
            name: weight for name, (_, weight) in losses.items()
        }

    def forward(self, **inputs):
        total = None
        metrics = {}

        for name, loss_fn in self.losses.items():
            loss_value, loss_metrics = loss_fn(**inputs)

            weighted = self.weights[name] * loss_value

            if total is None:
                total = weighted
            else:
                total = total + weighted

            metrics[f"{name}/loss"] = loss_value.detach()
            for k, v in loss_metrics.items():
                metrics[f"{name}/{k}"] = v.detach()

        return total, metrics

class MultitaskLoss(BaseLoss):
    def __init__(self, **tasks: BaseLoss):
        """
        tasks:
            seg=CompositeLoss(...)
            interp=CompositeLoss(...)
            synth=CompositeLoss(...)
        """
        super().__init__()
        self.tasks = nn.ModuleDict(tasks)

    def forward(self, batch: dict):
        total = None
        metrics = {}

        for name, loss_fn in self.tasks.items():
            if name not in batch:
                continue

            loss_value, task_metrics = loss_fn(**batch[name])

            if total is None:
                total = loss_value
            else:
                total = total + loss_value

            metrics[f"{name}/total"] = loss_value.detach()
            for k, v in task_metrics.items():
                metrics[f"{name}/{k}"] = v

        return total, metrics
