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


def prepare_synth_inputs(batch, device, num_classes: int):
    # segmentation label (middle frame)
    seg = (
        F.one_hot(batch.labels[1].squeeze(1).long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
    )

    # class id (assumed scalar per batch)
    # e.g. batch.class_id or batch.labels[1] contains class index
    input_class = batch.class_index[0].to(device)
    if torch.is_tensor(input_class):
        input_class = int(input_class.item())

    return seg, input_class

def _get_spade_input(batch, device):
    """Converts your Batch object into the dict SPADE expects."""
    # Assumption: batch.labels[1] is the semantic map for the synthesizer
    # Assumption: batch.images[1] is the ground truth target image
    data = {
        'label': batch.labels[1].to(device), 
        'image': batch.images[1].to(device),
        'instance': 0, # Add instance map here if you have it in batch
        'feat': 0      # Add features here if you have them
    }
    return data



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

def generator_step(synth, optimizer_G, data):
    optimizer_G.zero_grad()
    g_losses, generated = synth(data, mode='generator')
    g_loss = sum(g_losses.values()).mean()
    g_loss.backward()
    optimizer_G.step()
    return generated, g_loss

def discriminator_step(synth, optimizer_D, data):
    optimizer_D.zero_grad()
    d_losses = synth(data, mode='discriminator')
    d_loss = sum(d_losses.values()).mean()
    d_loss.backward()
    optimizer_D.step()
    return d_loss

def run_synth_gan(synth, optimizers, batch, context, training=True):
    """Orchestrates the GAN training steps."""
    if not synth:
        return None, 0.0, 0.0

    spade_data = _get_spade_input(batch, context.device)
    
    if training:
        d_loss = discriminator_step(synth, optimizers['disc'], spade_data)
        
        generated, g_loss = generator_step(synth, optimizers['synth'], spade_data)
        
        return generated, g_loss, d_loss
    else:
        with torch.no_grad():
            generated = synth(spade_data, mode='inference')
            return generated, 0.0, 0.0
