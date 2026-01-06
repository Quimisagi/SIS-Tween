import torch
import torch.nn.functional as F

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

def _prepare_synth_input(batch, device, num_classes=6):
    label = _prepare_label(batch.labels[1], device, num_classes)
    image0 = batch.images[0].to(device)
    image2 = batch.images[2].to(device)
    return label, image0, image2

def _segmentator_step(seg, loss_fn, image, label, device, weights=None):
    seg_img = image.to(device)
    seg_label = _prepare_label(label, device)

    assert seg_img.ndim == 4
    assert seg_label.ndim == 4
    assert seg_label.dtype == torch.float32

    seg_output = seg(seg_img)

    loss_ce = loss_fn.ce(seg_output, seg_label)
    loss_dice = loss_fn.dice(seg_output, seg_label)
    loss_focal = loss_fn.focal(seg_output, seg_label.argmax(dim=1))

    if weights:
        loss_ce *= weights.get('ce', 1.0)
        loss_dice *= weights.get('dice', 1.0)
        loss_focal *= weights.get('focal', 1.0)

    loss_total = loss_ce + loss_dice
    assert torch.isfinite(loss_total)

    return seg_output, loss_total

def run_segmentator(
    seg,
    loss_fn,
    batch,
    device,
    optimizer,
    weights,
    training: bool = True,
):
    generated = []
    total_loss = 0.0

    if training:
        seg.train()
    else:
        seg.eval()

    torch.set_grad_enabled(training)

    for img, lbl in zip(batch.images, batch.labels):
        if training:
            optimizer.zero_grad(set_to_none=True)

        output, loss_val = _segmentator_step(
            seg, loss_fn, img, lbl, device, weights
        )

        if training:
            loss_val.backward()
            optimizer.step()

        generated.append(output.detach())
        total_loss += loss_val.item()

    return generated, total_loss / len(batch.images)

def _interpolator_step(interp, loss_fn, batch, device, weights=None, num_classes=6):
    input_a = _prepare_interp_input(batch.labels[0], batch.images[0], device, num_classes)
    input_b = _prepare_interp_input(batch.labels[2], batch.images[2], device, num_classes)
    target = _prepare_label(batch.labels[1], device)

    generated = interp(input_a, input_b)
    assert generated.shape == target.shape

    loss_ce = loss_fn.ce(generated, target.argmax(dim=1))
    loss_dice = loss_fn.dice(generated, target)
    loss_focal = loss_fn.focal(generated, target.argmax(dim=1))

    if weights:
        loss_ce *= weights.get('ce', 1.0)
        loss_dice *= weights.get('dice', 1.0)
        loss_focal *= weights.get('focal', 1.0)

    loss_total = loss_ce + loss_dice
    assert torch.isfinite(loss_total)

    return generated, loss_total

def run_interpolator(
    interp,
    loss_fn,
    batch,
    device,
    optimizer,
    weights,
    training: bool = True,
):
    if training:
        interp.train()
    else:
        interp.eval()

    torch.set_grad_enabled(training)

    if training:
        optimizer.zero_grad(set_to_none=True)

    generated, loss_val = _interpolator_step(
        interp, loss_fn, batch, device, weights
    )

    if training:
        loss_val.backward()
        optimizer.step()

    return generated.detach(), loss_val.item()

def _synthesizer_step(
    synthesizer,
    loss_fn,
    batch,
    device,
    num_classes=6,
    weights=None,
):
    label, image0, image2 = _prepare_synth_input(batch, device, num_classes)

    generated = synthesizer(image0, image2, label)
    target = batch.images[1].to(device)
    loss_perceptual = loss_fn.perceptual(generated, target)



    if weights is not None:
        loss_perceptual = loss_perceptual * weights.get("perceptual", 1.0)

    loss_total = loss_perceptual

    assert torch.isfinite(loss_total).all(), "Non-finite loss detected"

    return generated, loss_total


def run_synthesizer(
    synthesizer,
    loss_fn,
    batch,
    device,
    optimizer,
    weights,
    training: bool = True,
    num_classes: int = 6,
):

    if training:
        synthesizer.train()
    else:
        synthesizer.eval()

    torch.set_grad_enabled(training)

    if training:
        optimizer.zero_grad(set_to_none=True)

    generated, loss_val = _synthesizer_step(
        synthesizer,
        loss_fn,
        batch,
        device,
        num_classes,
        weights,
    )
    if training:
        loss_val.backward()
        optimizer.step()



    return generated, loss_val.item()
