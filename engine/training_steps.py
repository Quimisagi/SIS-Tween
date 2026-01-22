import torch
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------


def prepare_label(label, device, num_classes: int):
    label = label.to(device)
    return (
        F.one_hot(label.squeeze(1).long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )

def prepare_interp_input(label, image, device, num_classes: int):
    label = prepare_label(label, device, num_classes)
    image = image.to(device)
    return torch.cat([image, label], dim=1)



def prepare_interp_input_from_seg(seg_onehot, image):
    """
    seg_onehot: [B, C, H, W]
    image: [B, 3, H, W]
    """
    return torch.cat([image, seg_onehot], dim=1)


# -----------------------------
# Segmentator
# -----------------------------


def segmentator_step(
    model,
    loss_fn,
    image,
    label,
    device,
    num_classes: int,
):
    image = image.to(device)
    target = prepare_label(label, device, num_classes)  

    output = model(image)

    # MultitaskLoss expects a batch dict
    loss, metrics = loss_fn(
        batch={
            "seg": {
                "pred": output,
                "target": target,
            }
        }
    )

    assert torch.isfinite(loss)
    return output, loss


def run_segmentator(
    model,
    loss_fn,
    batch,
    device,
    optimizer: torch.optim.Optimizer | None,
    training: bool = True,
    num_classes: int = 6,
):
    model.train(training)

    grad_ctx = torch.enable_grad() if training else torch.no_grad()

    outputs = []
    total_loss = 0.0

    with grad_ctx:
        for img, lbl in zip(batch.images, batch.labels):
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            out, loss = segmentator_step(model, loss_fn, img, lbl, device, num_classes)

            if training and optimizer is not None:
                loss.backward()
                optimizer.step()

            outputs.append(out.detach())
            total_loss += loss.item()

    return outputs, total_loss / len(batch.images)



# -----------------------------
# Interpolator
# -----------------------------


def interpolator_step(
    model,
    loss_fn,
    batch,
    device,
    num_classes: int,
):
    assert len(batch.images) == 3, "Batch must contain exactly 3 images for interpolation."
    input_a = prepare_interp_input(
        batch.labels[0], batch.images[0], device, num_classes
    )
    input_b = prepare_interp_input(
        batch.labels[2], batch.images[2], device, num_classes
    )
    target = prepare_label(batch.labels[1], device, num_classes)

    output = model(input_a, input_b)


    loss, metrics = loss_fn(
        batch={
            "interp": {
                "pred": output,
                "target": target,
            }
        }
    )

    assert torch.isfinite(loss)
    return output, loss


def run_interpolator(
    model,
    loss_fn,
    batch,
    device,
    optimizer: torch.optim.Optimizer | None,
    training: bool = True,
    num_classes: int = 6,
):
    model.train(training)

    grad_ctx = torch.enable_grad() if training else torch.no_grad()

    with grad_ctx:
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        output, loss = interpolator_step(model, loss_fn, batch, device, num_classes)

        if training and optimizer is not None:
            loss.backward()
            optimizer.step()

    return output.detach(), loss.item()

