import torch.nn.functional as F

def multiclass_dice_loss(logits, target, eps=1e-6):
    """
    logits: (B, C, H, W) — raw model output (before softmax)
    target: (B, H, W)    — class indices
    """
    num_classes = logits.shape[1]

    probs = F.softmax(logits, dim=1)

    # # target → one-hot
    # target = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
    # target = target.permute(0, 3, 1, 2).float()           # (B, C, H, W)

    # flatten spatial dims
    probs = probs.reshape(probs.size(0), num_classes, -1)
    target = target.reshape(target.size(0), num_classes, -1)

    intersection = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1)

    dice_per_class = (2 * intersection + eps) / (union + eps)

    return 1 - dice_per_class.mean()
