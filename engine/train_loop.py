import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from utils.dice_score import dice_score_multiclass
from utils.visualization import samples_comparison, plot_losses

def _prepare_label(label, device, num_classes=6):
    label = label.to(device)
    return F.one_hot(
        label.squeeze(1).long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

def _prepare_interp_input(label, image, device, num_classes=6):
    label = _prepare_label(label, device, num_classes)
    image = image.to(device)
    return torch.cat([image, label], dim=1)

def _train_segmentator(seg, loss, optimizers, images, labels, device, weights=None, is_validation=False):
    generated = []
    loss_seg_total = 0.0

    for i in range(len(images)):
        seg_img = images[i].to(device)
        seg_label = _prepare_label(labels[i], device)

        assert seg_img.ndim == 4
        assert seg_label.ndim == 4
        assert seg_label.dtype == torch.float32

        if not is_validation and optimizers:
            optimizers['seg'].zero_grad()

        seg_output = seg(seg_img)

        loss_ce = loss.ce(seg_output, seg_label)
        loss_dice = loss.dice(seg_output, seg_label)
        if weights:
            loss_ce *= weights['seg'].get('ce', 1.0)
            loss_dice *= weights['seg'].get('dice', 1.0)

        loss_seg = loss_ce + loss_dice
        assert torch.isfinite(loss_seg)

        if not is_validation and optimizers:
            loss_seg.backward()
            optimizers['seg'].step()

        generated.append(seg_output.detach())
        loss_seg_total += loss_seg.item()

    return generated, loss_seg_total / len(images)

def _train_interpolator(interp, loss, optimizers, images, labels, device, weights=None, is_validation=False):
    if not is_validation and optimizers:
        optimizers['interp'].zero_grad()

    input_a = _prepare_interp_input(labels[0], images[0], device)
    input_b = _prepare_interp_input(labels[2], images[2], device)
    target = _prepare_label(labels[1], device)

    generated = interp(input_a, input_b)
    assert generated.shape == target.shape

    loss_ce = loss.ce(generated, target.argmax(dim=1))
    loss_dice = loss.dice(generated, target)
    if weights:
        loss_ce *= weights['interp'].get('ce', 1.0)
        loss_dice *= weights['interp'].get('dice', 1.0)

    loss_interp = loss_ce + loss_dice
    assert torch.isfinite(loss_interp)

    if not is_validation and optimizers:
        loss_interp.backward()
        optimizers['interp'].step()

    return generated.detach(), loss_interp.item()

def validate(seg, interp, loss, dataloader, device, writer=None, epoch=0, logger=None):
    seg.eval()
    interp.eval()
    total_loss_seg = 0.0
    total_loss_interp = 0.0
    total_dice_seg = 0.0

    n_batches = len(dataloader)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data['images']
            labels = data['labels']

            seg_output, loss_seg = _train_segmentator(seg, loss, None, images, labels, device, weights=None, is_validation=True)
            interp_output, loss_interp = _train_interpolator(interp, loss, None, images, labels, device, weights=None, is_validation=True)

            total_loss_seg += loss_seg
            total_loss_interp += loss_interp

            for i in range(len(seg_output)):
                seg_output[i] = seg_output[i].detach()
                labels[i] = labels[i].to(device).squeeze(1).long()
                dice = dice_score_multiclass(seg_output[i], labels[i])
                total_dice_seg += dice

            if writer is not None and i % 50 == 0:
                samples_comparison(writer, logger, images, labels, seg_output, interp_output, epoch, tag="val_samples")

    avg_loss_seg = total_loss_seg / n_batches
    avg_loss_interp = total_loss_interp / n_batches
    avg_dice_seg = total_dice_seg / n_batches / 3

    if logger:
        logger.info(f"[Validation] Seg_Loss={avg_loss_seg:.4f}, Interp_Loss={avg_loss_interp:.4f}")

    if writer:
        plot_losses(writer, logger, {'Segmentation': avg_loss_seg, 'Interpolation': avg_loss_interp}, epoch * n_batches)

    return avg_loss_seg, avg_loss_interp, avg_dice_seg

def train_loop(seg, interp, loss, optimizers, dataloader, device, writer, logger, weights, epochs=50, segmentator_score_threshold=0.1):
    #TODO: Too many arguments, refactor
    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        seg.train()
        interp.train()

        for i, data in enumerate(dataloader):
            images = data['images']
            labels = data['labels']

            seg_output, loss_seg = _train_segmentator(seg, loss, optimizers, images, labels, device, weights)
            interp_output, loss_interp = _train_interpolator(interp, loss, optimizers, images, labels, device, weights)

            if writer and i % 100 == 0:
                logger.info(f"[Step:{i}]: Seg_Loss={loss_seg:.4f}, Interp_Loss={loss_interp:.4f}")
                plot_losses(writer, logger, {'Segmentation': loss_seg, 'Interpolation': loss_interp}, epoch * len(dataloader) + i)
                samples_comparison(writer, logger, images, labels, seg_output, interp_output, epoch, tag="train_samples")

        val_loss_seg, val_loss_interp, dice_score_seg = validate(seg, interp, loss, dataloader, device, writer, epoch, logger)
        logger.info(f"[Epoch:{epoch + 1}/{epochs}] Validation Seg_Loss={val_loss_seg:.4f}, Interp_Loss={val_loss_interp:.4f}, Dice_Score={dice_score_seg:.4f}")

        if dice_score_seg > segmentator_score_threshold:
            logger.info(f"Dice score {dice_score_seg:.4f} exceeded threshold {segmentator_score_threshold:.4f}.")

