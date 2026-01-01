import torch
import torch.nn as nn
import torch.nn.functional as F

def _prepare_label(label, device, num_classes=7):
    label = label.to(device)
    return F.one_hot(
        label.squeeze(1).long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

def _prepare_interp_input(label, image, device, num_classes=7):
    label = _prepare_label(label, device, num_classes)
    image = image.to(device)
    return torch.cat([image, label], dim=1)


def _train_segmentator(seg, loss, optimizers, images, labels, device, weights):
    # ---- Segmentation training ----
    for i in range(3):
        seg_img = images[i].to(device)
        seg_label = _prepare_label(labels[i], device)

        optimizers['seg'].zero_grad()
        seg_output = seg(seg_img)

        loss_ce = loss.ce(seg_output, seg_label) * weights['seg']['ce']
        loss_dice = loss.dice(seg_output, seg_label) * weights['seg']['dice']
        loss_seg = loss_ce + loss_dice

        loss_seg.backward()
        optimizers['seg'].step()


def _train_interpolator(interp, loss, optimizers, images, labels, device, weights):
    # ---- Interpolator training ----
    optimizers['interp'].zero_grad()

    input_a = _prepare_interp_input(labels[0], images[0], device)
    input_b = _prepare_interp_input(labels[2], images[2], device)
    target = _prepare_label(labels[1], device)

    generated = interp(input_a, input_b)

    loss_ce = loss.ce(
        generated,
        target.argmax(dim=1)
    ) * weights['interp']['ce']

    loss_dice = loss.dice(
        generated,
        target
    ) * weights['interp']['dice']

    loss_interp = loss_ce + loss_dice

    loss_interp.backward()
    optimizers['interp'].step()


def train_loop(seg, interp, loss, optimizers, dataloader, device, writer, logger, weights, epochs=50):
    seg.train()
    interp.train()

    for epoch in range(epochs):
        logger.info(f"[Epoch:{epoch + 1}/{epochs}]")
        for data in dataloader:
            images = data['images']
            labels = data['labels']
            if epoch == 0:
                logger.debug(f"Images length: {len(images)}")
                logger.debug(f"Labels length: {len(labels)}")

            _train_segmentator(seg, loss, optimizers, images, labels, device, weights)
            _train_interpolator(interp, loss, optimizers, images, labels, device, weights)

