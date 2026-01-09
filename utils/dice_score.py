import torch
import torch.nn.functional as F

def dice_score(preds, labels, smooth=1e-6):
    preds_class = preds.argmax(dim=1)
    preds_flat = preds_class.view(-1)
    labels_flat = labels.view(-1)
    
    intersection = (preds_flat == labels_flat).float().sum()
    return (2. * intersection + smooth) / (preds_flat.numel() + labels_flat.numel() + smooth)

def dice_score_multiclass(preds, labels, num_classes=None, eps=1e-6):
    """
    Computes the average Dice score over all classes.
    preds: (B, C, H, W) logits from the model
    labels: (B, H, W) integer class labels
    """

    assert preds.dim() == 4, f"Expected preds to have 4 dimensions (B, C, H, W), got {preds.dim()}"
    assert labels.dim() == 3, f"Expected labels to have 3 dimensions (B, H, W), got {labels.dim()}"
    if num_classes is None:
        num_classes = preds.shape[1]

    labels = labels.to(preds.device)

    # Convert labels to one-hot
    labels_onehot = F.one_hot(labels, num_classes=num_classes)  # (B, H, W, C)
    labels_onehot = labels_onehot.permute(0, 3, 1, 2).float()   # (B, C, H, W)

    # Get predicted class
    preds_class = preds.argmax(dim=1)                           # (B, H, W)
    preds_onehot = F.one_hot(preds_class, num_classes=num_classes)
    preds_onehot = preds_onehot.permute(0, 3, 1, 2).float()     # (B, C, H, W)

    # Flatten spatial dims
    preds_flat = preds_onehot.reshape(preds_onehot.size(0), num_classes, -1)
    labels_flat = labels_onehot.reshape(labels_onehot.size(0), num_classes, -1)

    intersection = (preds_flat * labels_flat).sum(dim=-1)
    union = preds_flat.sum(dim=-1) + labels_flat.sum(dim=-1)
    dice_per_class = (2 * intersection + eps) / (union + eps)

    return dice_per_class.mean().item()  # scalar
