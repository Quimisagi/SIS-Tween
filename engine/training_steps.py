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

def prepare_synth_gan_inputs(batch, device, num_classes: int):
    # segmentation label (middle frame)
    seg = (
        F.one_hot(batch.labels[1].squeeze(1).long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
        )
    input_class = batch.class_index[0].to(device)
    if torch.is_tensor(input_class):
        input_class = int(input_class.item())

    edge = batch.edges[1].to(device)

    return seg, edge, input_class


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


# -----------------------------
# Synthesizer
# -----------------------------

def synthesizer_G_step(
    generator,
    discriminator,
    loss_fn,
    batch,
    device,
    num_classes: int,
):
    seg, edge, input_class = prepare_synth_gan_inputs(batch, device, num_classes)

    fake = generator(seg, input_class)
    output_D = discriminator(fake)

    # Build batch dict for GAN loss
    loss_batch = {
        'synth': {
            'disc_pred': output_D,
            'seg': seg,  
            'label_canny': edge,
            'for_real': True,
            'pred': fake,
            'target': batch.images[1].to(device)
        }
    }

    loss_G, metrics_G = loss_fn(loss_batch)

    return fake, loss_G, metrics_G


def synthesizer_D_step(
    generator,
    discriminator,
    loss_fn,
    batch,
    device,
    num_classes: int,
):
    seg, edge, input_class = prepare_synth_gan_inputs(batch, device, num_classes)
    target = batch.images[1].to(device)

    with torch.no_grad():
        fake = generator(seg, input_class)

    # Real
    loss_batch_real = {
        'synth': {
            'disc_pred': discriminator(target),
            'seg': seg,  
            'label_canny': edge,
            'for_real': True
        }
    }
    loss_D_real, metrics_D_real = loss_fn(loss_batch_real)

    # Fake
    loss_batch_fake = {
        'synth': {
            'disc_pred': discriminator(fake.detach()),
            'seg': seg, 
            'label_canny': edge,
            'for_real': False
        }
    }
    loss_D_fake, metrics_D_fake = loss_fn(loss_batch_fake)

    loss_D = loss_D_real + loss_D_fake
    metrics_D = {}
    for k, v in metrics_D_real.items():
        metrics_D[f'real_{k}'] = v
    for k, v in metrics_D_fake.items():
        metrics_D[f'fake_{k}'] = v

    return loss_D, metrics_D

def run_synthesizer_gan(
    generator,
    discriminator,
    loss_fn,
    batch,
    device,
    optimizer_G: torch.optim.Optimizer | None,
    optimizer_D: torch.optim.Optimizer | None,
    training: bool = True,
    num_classes: int = 6,
):
    generator.train(training)
    discriminator.train(training)

    grad_ctx = torch.enable_grad() if training else torch.no_grad()

    loss_G_total = 0.0
    loss_D_total = 0.0

    with grad_ctx:
        # Update Discriminator
        if training and optimizer_D is not None:
            optimizer_D.zero_grad(set_to_none=True)

        loss_D, metrics_D = synthesizer_D_step(
            generator, discriminator, loss_fn, batch, device, num_classes
        )

        if training and optimizer_D is not None:
            loss_D.backward()
            optimizer_D.step()

        loss_D_total += loss_D.item()

        # Update Generator
        if training and optimizer_G is not None:
            optimizer_G.zero_grad(set_to_none=True)

        fake, loss_G, metrics_G = synthesizer_G_step(
            generator, discriminator, loss_fn, batch, device, num_classes
        )

        if training and optimizer_G is not None:
            loss_G.backward()
            optimizer_G.step()

        loss_G_total += loss_G.item()

    return fake, loss_G_total, loss_D_total


def joint_seg_interp_synth_step(
    segmentator,
    interpolator,
    synthesizer_G,
    synthesizer_D,
    loss_fn,
    batch,
    device,
    num_classes: int,
    optimizer_seg=None,
    optimizer_interp=None,
    optimizer_G=None,
    optimizer_D=None,
    training: bool = True,
):
    """
    Flow:
      img0,img1,img2
        ↓
      segmentator(img0,img1,img2)
        ↓ (seg predictions)
      interpolator(seg0+img0 , seg2+img2) → seg1_hat
        ↓
      synthesizer(seg1_hat) → img1_hat
        ↓
      single loss_fn backward through everything
    """

    segmentator.train(training)
    interpolator.train(training)
    synthesizer_G.train(training)
    synthesizer_D.train(training)

    grad_ctx = torch.enable_grad() if training else torch.no_grad()

    with grad_ctx:
        # -----------------------------
        # Zero grads
        # -----------------------------
        if training:
            for opt in [optimizer_seg, optimizer_interp, optimizer_G, optimizer_D]:
                if opt is not None:
                    opt.zero_grad(set_to_none=True)

        # -----------------------------
        # 1) SEGMENTATION (all frames)
        # -----------------------------
        imgs = [img.to(device) for img in batch.images]
        gt_labels = [lbl.to(device) for lbl in batch.labels]

        seg_logits = [segmentator(img) for img in imgs]
        seg_preds = [torch.softmax(s, dim=1) for s in seg_logits]

        seg_targets = [
            prepare_label(lbl, device, num_classes) for lbl in gt_labels
        ]

        # -----------------------------
        # 2) INTERPOLATION (use seg predictions)
        # -----------------------------
        interp_in_a = prepare_interp_input_from_seg(seg_preds[0], imgs[0])
        interp_in_b = prepare_interp_input_from_seg(seg_preds[2], imgs[2])

        seg_mid_hat = interpolator(interp_in_a, interp_in_b)

        # -----------------------------
        # 3) SYNTHESIS (use interpolated seg)
        # -----------------------------
        input_class = batch.class_index[0].to(device)
        if torch.is_tensor(input_class):
            input_class = int(input_class.item())

        fake_mid = synthesizer_G(seg_mid_hat, input_class)

        # -----------------------------
        # 4) DISCRIMINATOR
        # -----------------------------
        real_mid = imgs[1]
        edge_mid = batch.edges[1].to(device)

        disc_real = synthesizer_D(real_mid)
        disc_fake = synthesizer_D(fake_mid)

        # -----------------------------
        # 5) SINGLE LOSS
        # -----------------------------
        loss_batch = {
            "seg": {
                "pred": seg_logits[1],
                "target": seg_targets[1],
            },
            "interp": {
                "pred": seg_mid_hat,
                "target": seg_targets[1],
            },
            "synth_G": {
                "disc_pred": disc_fake,
                "seg": seg_mid_hat,
                "label_canny": edge_mid,
                "for_real": True,
                "pred": fake_mid,
                "target": real_mid,
            },
            "synth_D_real": {
                "disc_pred": disc_real,
                "seg": seg_targets[1],
                "label_canny": edge_mid,
                "for_real": True,
            },
            "synth_D_fake": {
                "disc_pred": disc_fake.detach(),
                "seg": seg_targets[1],
                "label_canny": edge_mid,
                "for_real": False,
            },
        }

        total_loss, metrics = loss_fn(loss_batch)

        assert torch.isfinite(total_loss)

        # -----------------------------
        # 6) BACKWARD (once)
        # -----------------------------
        if training:
            total_loss.backward()

            for opt in [optimizer_seg, optimizer_interp, optimizer_G, optimizer_D]:
                if opt is not None:
                    opt.step()

    return {
        "fake_mid": fake_mid.detach(),
        "loss": total_loss.item(),
        "metrics": metrics,
    }
