import visualization

def log_step(self, stage, losses, batch=None, outputs=None):
    if not self.context.writer or self.global_step % 300 != 0:
        return
    loss_str = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
    self.context.logger.info(
        f"[{stage}][Epoch:{self.epoch+1}/{self.opt.epochs}][Step:{self.global_step}] {loss_str}"
    )
    if "Segmentation" in losses:
        visualization.plot(
            self.context,
            {"train": losses["Segmentation"]},
            self.global_step,
            "loss/segmentation",
        )
    if "Interpolation" in losses:
        visualization.plot(
            self.context,
            {"train": losses["Interpolation"]},
            self.global_step,
            "loss/interpolation",
        )
    if "Synthesis_G" in losses:
        visualization.plot(
            self.context,
            {"train": losses["Synthesis_G"]},
            self.global_step,
            "loss/synthesis_G",
        )
    if "Synthesis_D" in losses:
        visualization.plot(
            self.context,
            {"train": losses["Synthesis_D"]},
            self.global_step,
            "loss/synthesis_D",
        )

    outputs = {
            k: self.safe_detach(v) for k, v in (outputs or {}).items()
    }

    if batch is not None and outputs is not None:
        visualization.samples_comparison(
            self.context,
            batch.images,
            batch.labels,
            outputs,
            self.global_step,
            tag="samples",
        )

    def log_validation(self, metrics: dict):
        loss = metrics["loss"]
        dice = metrics["dice"]
        psnr = metrics["psnr"]

        if self.context.logger:
            self.context.logger.info(
                "[Validation] "
                f"Seg_Loss={loss['seg']:.4f}, "
                f"Interp_Loss={loss['interp']:.4f}, "
                f"Synth_G_Loss={loss['G']:.4f}, "
                f"Synth_D_Loss={loss['D']:.4f}, "
                f"Dice_Seg={dice['seg']:.4f}, "
                f"Dice_Interp={dice['interp']:.4f}"
            )

        if self.context.writer:
            visualization.plot(
                self.context,
                {"segmentation": dice["seg"], "interpolation": dice["interp"]},
                self.global_step,
                tag="dice",
            )

            visualization.plot(
                self.context,
                {"validation": loss["seg"]},
                self.global_step,
                tag="loss/segmentation",
            )

            visualization.plot(
                self.context,
                {"validation": loss["G"]},
                self.global_step,
                tag="loss/synthesis_G",
            )

            visualization.plot(
                self.context,
                {"validation": loss["D"]},
                self.global_step,
                tag="loss/synthesis_D",
            )

            visualization.plot(
                self.context,
                {"validation": loss["interp"]},
                self.global_step,
                tag="loss/interpolation",
            )

            visualization.plot(
                self.context,
                {"synth_psnr": psnr["synth"]},
                self.global_step,
                tag="psnr",
            )

