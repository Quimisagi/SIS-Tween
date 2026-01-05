import torch
import torch.nn as nn
import torch.nn.functional as F

def multiclass_dice_loss(logits, target, eps=1e-6):
    """
    logits: (B, C, H, W) — raw model output (before softmax)
    target: (B, H, W)    — class indices
    """
    num_classes = logits.shape[1]

    probs = F.softmax(logits, dim=1)

    # flatten spatial dims
    probs = probs.reshape(probs.size(0), num_classes, -1)
    target = target.reshape(target.size(0), num_classes, -1)

    intersection = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1)

    dice_per_class = (2 * intersection + eps) / (union + eps)

    return 1 - dice_per_class.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
):
    ce = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        reduction="none",
    )

    pt = torch.exp(-ce)
    loss = (1.0 - pt) ** gamma * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class Loss:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()

    def ce(self, p, t):
        return self.ce(p, t)

    def dice(self, p, t):
        return multiclass_dice_loss(p, t)

    def focal(self, p, t):
        return focal_loss(p, t)
