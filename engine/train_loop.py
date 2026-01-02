import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _train_segmentator(seg, loss, optimizers, images, labels, device, weights, is_validation=False):
    generated = []
    for i in range(3):
        seg_img = images[i].to(device)
        seg_label = _prepare_label(labels[i], device)

        assert seg_img.ndim == 4  # [B, C, H, W]
        assert seg_label.ndim == 4
        assert seg_label.dtype == torch.float32

        optimizers['seg'].zero_grad()
        seg_output = seg(seg_img)

        loss_ce = loss.ce(seg_output, seg_label) * weights['seg']['ce']
        loss_dice = loss.dice(seg_output, seg_label) * weights['seg']['dice']

        loss_seg = loss_ce + loss_dice
        assert torch.isfinite(loss_seg)

        loss_seg.backward()
        optimizers['seg'].step()

        generated.append(seg_output.detach())
    return generated, loss_seg


def _train_interpolator(interp, loss, optimizers, images, labels, device, weights, is_validation=False):
    optimizers['interp'].zero_grad()

    input_a = _prepare_interp_input(labels[0], images[0], device)
    input_b = _prepare_interp_input(labels[2], images[2], device)
    assert input_a.shape == input_b.shape

    target = _prepare_label(labels[1], device)
    generated = interp(input_a, input_b)
    assert generated.shape == target.shape

    loss_ce = loss.ce(
        generated,
        target.argmax(dim=1)
    ) * weights['interp']['ce']

    loss_dice = loss.dice(
        generated,
        target
    ) * weights['interp']['dice']

    loss_interp = loss_ce + loss_dice
    assert torch.isfinite(loss_interp)

    loss_interp.backward()
    optimizers['interp'].step()

    return generated, loss_interp

def train_loop(seg, interp, loss, optimizers, dataloader, device, writer, logger, weights, epochs=50):
    seg.train()
    interp.train()
    for epoch in range(epochs):
        logger.info(f"[Epoch:{epoch + 1}/{epochs}]")
        for i, data in enumerate(dataloader):
            images = data['images']
            labels = data['labels']
            if epoch == 0:
                assert 'images' in data and 'labels' in data
                assert len(images) == 3
                assert len(labels) == 3

            seg_output, loss_seg= _train_segmentator(seg, loss, optimizers, images, labels, device, weights)
            interp_output, loss_interp = _train_interpolator(interp, loss, optimizers, images, labels, device, weights)

            if writer is not None and i % 100 == 0:
                logger.info(f"[Step:{i}]: Seg_Loss={loss_seg.item():.4f}, Interp_Loss={loss_interp.item():.4f}")
                plot_losses(
                    writer,
                    logger,
                    {
                        'Segmentation': loss_seg.item(),
                        'Interpolation': loss_interp.item()
                    },
                    epoch * len(dataloader) + i
                )
                samples_comparison(
                    writer,
                    logger,
                    gt_images=images,
                    gt_labels=labels,
                    seg_labels=seg_output,
                    interp_label=interp_output,
                    epoch=epoch,
                    tag="train_samples"
                )

