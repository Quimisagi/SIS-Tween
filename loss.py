import torch
import torch.nn.functional as F

def multiclass_dice_loss(logits, target, eps=1e-6):
    """
    logits: (B, C, H, W) — raw model output (before softmax)
    target: (B, C, H, W) — one-hot segmentation mask
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    
    # flatten tensors
    probs = probs.reshape(probs.size(0), num_classes, -1)
    target = target.reshape(target.size(0), num_classes, -1)
    
    # compute Dice per class
    intersection = (probs * target).sum(-1)
    union = probs.sum(-1) + target.sum(-1)
    
    dice_per_class = (2 * intersection + eps) / (union + eps)
    
    # return 1 - mean Dice
    return 1 - dice_per_class.mean()
