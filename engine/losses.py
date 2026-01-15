import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
import inspect

def filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn.forward)
    return {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
def get_class_balancing(opt, input, label):

    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map

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
    def __init__(self, weight: torch.Tensor | None = None, reduction: str = "mean", apply_class_balancing: bool = True):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.apply_class_balancing = apply_class_balancing

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor | None = None):
        """
        pred:  (B, C, H, W)
        target: (B, H, W)
        weight_map: optional per-pixel weights for class balancing (B, 1, H, W)
        """
        if self.apply_class_balancing and weight_map is not None:
            ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
            loss = torch.mean(ce * weight_map[:, 0, :, :])
        else:
            loss = F.cross_entropy(pred, target, weight=self.weight, reduction=self.reduction)
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
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36]
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

class L1Loss(BaseLoss):
    def __init__(self, reduction: str = "mean"):
        """
        reduction: "mean" | "sum" | "none"
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, C, H, W)
        Expected to be in the SAME space and range
        (e.g. both in [0,1] or both in [-1,1])
        """
        loss = F.l1_loss(
            pred,
            target,
            reduction=self.reduction,
        )

        # For logging consistency, always report a scalar metric
        if self.reduction == "none":
            metric = loss.mean()
        else:
            metric = loss

        return loss, {"l1": metric}

class OASISGanLoss(BaseLoss):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device

    def get_n1_target(self, disc_pred, seg, target_is_real):
        targets = self.get_target_tensor(disc_pred, target_is_real)
        num_of_classes = seg.shape[1]
        integers = torch.argmax(seg, dim=1)
        targets = targets[:, 0, :, :] * num_of_classes
        integers += targets.long()
        integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
        return integers

    def get_target_tensor(self, disc_pred, target_is_real):
        fill_value = 1.0 if target_is_real else 0.0
        return torch.full_like(disc_pred, fill_value, device=self.device, requires_grad=False)

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask * output_D_real + (1-mask) * output_D_fake
        return F.mse_loss(mixed_D_output, output_D_mixed)

    def forward(self, disc_pred: torch.Tensor, seg: torch.Tensor, label_canny: torch.Tensor, for_real: bool):
        # Split discriminator outputs: classes vs edges
        input_canny = disc_pred[:, -1:, :, :]
        input_classes = disc_pred[:, :-1, :, :]

        # Compute per-pixel class balancing
        weight_map = get_class_balancing(self.opt, input_classes, seg)

        # n+1 loss targets
        target = self.get_n1_target(input_classes, seg, for_real)
        loss = F.cross_entropy(input_classes, target, reduction='none')

        
        label_canny = (label_canny > 0).float()


        # Edge loss
        if for_real:
            canny_loss = F.binary_cross_entropy_with_logits(input_canny, label_canny)
        else:
            target_canny = torch.zeros_like(label_canny, device=self.device)
            canny_loss = F.binary_cross_entropy_with_logits(input_canny, target_canny)

        # Apply class weighting to class loss (only for real)
        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
            canny_loss = torch.mean(canny_loss)
        else:
            loss = torch.mean(loss)
            canny_loss = torch.mean(canny_loss)

        loss = loss + 0.3 * canny_loss

        return loss, {'gan': loss, 'canny': canny_loss}
    

class CompositeLoss(BaseLoss):
    def __init__(self, losses: dict[str, tuple[BaseLoss, float]]):
        super().__init__()
        self.losses = nn.ModuleDict({name: loss for name, (loss, _) in losses.items()})
        self.weights = {name: weight for name, (_, weight) in losses.items()}

    def forward(self, **inputs):
        total = None
        metrics = {}

        for name, loss_fn in self.losses.items():
            filtered_inputs = filter_kwargs(loss_fn, inputs)

            # Skip loss if it has no valid inputs
            if len(filtered_inputs) == 0:
                continue

            loss_value, loss_metrics = loss_fn(**filtered_inputs)
            weighted = self.weights[name] * loss_value

            total = weighted if total is None else total + weighted

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
