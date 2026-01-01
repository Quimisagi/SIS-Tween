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

class Loss:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()

    def ce(self, p, t):
        return self.ce(p, t)

    def dice(self, p, t):
        return multiclass_dice_loss(p, t)
