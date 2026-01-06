import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights

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

def perceptual_loss(x, y, vgg):
    """
    Computes perceptual loss between x (fake) and y (real).
    Assumes images are in range [0, 1] or [-1, 1] and 4D tensors.
    """
    return F.l1_loss(vgg(x), vgg(y))


class Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:19].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def ce(self, p, t):
        return self.ce_loss(p, t)

    def dice(self, p, t):
        return multiclass_dice_loss(p, t)

    def focal(self, p, t):
        return focal_loss(p, t)

    def perceptual(self, p, t):
        # Ensure p and t are 3-channel RGB [B, 3, H, W]
        p_vgg = self.vgg(p)
        t_vgg = self.vgg(t)
        return torch.nn.functional.l1_loss(p_vgg, t_vgg)
