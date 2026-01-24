from utils import visualization

def safe_detach(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [t.detach().cpu() for t in x]
    return x.detach().cpu()


def log_step(context, opt, global_step, epoch, stage, losses, batch=None, outputs=None):
    if not context.writer or global_step % opt.freq_print != 0:
        return
    loss_str = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
    context.logger.info(
        f"[{stage}][Epoch:{epoch+1}/{opt.epochs}][Step:{global_step}] {loss_str}"
    )
    if "Segmentation" in losses:
        visualization.plot(
            context,
            {"train": losses["Segmentation"]},
            global_step,
            "loss/segmentation",
        )
    if "Interpolation" in losses:
        visualization.plot(
            context,
            {"train": losses["Interpolation"]},
            global_step,
            "loss/interpolation",
        )

    outputs = {
            k: safe_detach(v) for k, v in (outputs or {}).items()
    }

    if batch is not None and outputs is not None:
        visualization.samples_comparison(
            context,
            batch.images,
            batch.labels,
            outputs,
            global_step,
            tag="samples",
        )

def log_validation(context, global_step, metrics: dict):
    loss = metrics["loss"]
    dice = metrics["dice"]

    if context.logger:
        context.logger.info(
            "[Validation] "
            f"Seg_Loss={loss['seg']:.4f}, "
            f"Interp_Loss={loss['interp']:.4f}, "
            f"Dice_Seg={dice['seg']:.4f}, "
            f"Dice_Interp={dice['interp']:.4f}"
        )

    if context.writer:
        visualization.plot(
            context,
            {"segmentation": dice["seg"], "interpolation": dice["interp"]},
            global_step,
            tag="dice",
        )

        visualization.plot(
            context,
            {"validation": loss["seg"]},
            global_step,
            tag="loss/segmentation",
        )

        visualization.plot(
            context,
            {"validation": loss["interp"]},
            global_step,
            tag="loss/interpolation",
        )


