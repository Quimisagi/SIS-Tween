from torch.nn.parallel import DistributedDataParallel
import torch
from torch.utils.data import DistributedSampler

from models import Interpolator, Segmentator, Synthesizer, Discriminator
from utils import visualization
from utils.psnr import psnr
from .bundles import Batch
from .training_steps import run_segmentator, run_interpolator, run_synthesizer, run_synthesizer_gan
from collections import deque
from utils.dice_score import dice_score_multiclass


class Trainer:
    def __init__(self, loss_fn, dataloaders, opt, context):
        self.opt = opt
        self.context = context
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.device = context.device

        self.seg = (
            Segmentator(num_seg_classes=opt.semantic_nc).to(self.device)
            if "seg" in opt.active_models
            else None
        )
        self.interp = (
            Interpolator(sem_c=opt.semantic_nc, base_c=64).to(self.device)
            if "interp" in opt.active_models
            else None
        )
        self.synth = (
            Synthesizer(opt, context).to(self.device)
            if "synth" in opt.active_models
            else None
        )
        self.disc = (
            Discriminator(opt).to(self.device)
            if "synth" in opt.active_models
            else None
        )

        if opt.distributed_enabled and context.world_size > 1:
            if self.seg:
                self.seg = DistributedDataParallel(
                    self.seg, device_ids=[context.local_rank]
                )
            if self.interp:
                self.interp = DistributedDataParallel(
                    self.interp, device_ids=[context.local_rank]
                )
            if self.synth:
                self.synth = DistributedDataParallel(
                    self.synth, device_ids=[context.local_rank]
                )
            if self.disc:
                self.disc = DistributedDataParallel(
                    self.disc, device_ids=[context.local_rank]
                )

        self.optimizers = {}
        if self.seg:
            self.optimizers["seg"] = torch.optim.Adam(
                self.seg.parameters(), lr=opt.lr_seg
            )
        if self.interp:
            self.optimizers["interp"] = torch.optim.Adam(
                self.interp.parameters(), lr=opt.lr_interp
            )
        if self.synth:
            self.optimizers["synth"] = torch.optim.Adam(
                self.synth.parameters(), lr=opt.lr_synth
            )
        if self.disc:
            self.optimizers["disc"] = torch.optim.Adam(
                self.disc.parameters(), lr=opt.lr_d
            )

        self.schedulers = {}
        if self.seg:
            self.schedulers["seg"] = torch.optim.lr_scheduler.StepLR(
                self.optimizers["seg"], 10, 0.1
            )
        if self.interp:
            self.schedulers["interp"] = torch.optim.lr_scheduler.StepLR(
                self.optimizers["interp"], 10, 0.1
            )
        if self.synth:
            self.schedulers["synth"] = torch.optim.lr_scheduler.StepLR(
                self.optimizers["synth"], 10, 0.1
            )
        if self.disc:
            self.schedulers["disc"] = torch.optim.lr_scheduler.StepLR(
                self.optimizers["disc"], 10, 0.1
            )

        self.epoch = 0
        self.global_step = 0
        self.train_stage = 0

    def make_batch(self, data):
        """
        data: dict with keys "images" and "labels"
        """
        return Batch(images=data["images"], labels=data["labels"], edges=data["edges"], class_index=data["class_index"])

    def relative_improvement(self, values, eps: float = 1e-8) -> float:
        """
        Computes a *stability-aware* relative improvement over a sequence.

        The metric considers **all values**, not just first/last:

        1. Compute relative improvement from the mean of the first half
           to the mean of the second half
        2. Penalize non-stable sequences using normalized variance

        Returns:
            RI = improvement * stability

        Where:
            improvement = (mean_old - mean_new) / max(|mean_old|, eps)
            stability   = 1 / (1 + coeff_variation)
        """
        n = len(values)
        values = list(values)
        if n < 2:
            return 0.0

        # Split sequence
        mid = n // 2
        first = values[:mid]
        second = values[mid:]

        mean_old = sum(first) / len(first)
        mean_new = sum(second) / len(second)

        improvement = (mean_old - mean_new) / max(abs(mean_old), eps)

        # Stability: coefficient of variation over full window
        mean_all = sum(values) / n
        var = sum((v - mean_all) ** 2 for v in values) / n
        std = var ** 0.5
        coeff_variation = std / max(abs(mean_all), eps)

        stability = 1.0 / (1.0 + coeff_variation)

        return improvement * stability

    def replace_labels_with_segmentation(self, batch: Batch, seg_output):
        if seg_output is not None:
            batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]

    def replace_middle_with_interpolation(self, batch: Batch, interp_output):
        if interp_output is not None:
            batch.labels[1] = interp_output.detach().argmax(dim=1)

    def safe_detach(self, x):
        if x is None:
            return None
        if isinstance(x, list):
            return [t.detach().cpu() for t in x]
        return x.detach().cpu()


    def forward_seg(self, batch, optimizer=None, require_grad=True):
        if not self.seg:
            return None, 0.0
        return run_segmentator(
            self.seg, self.loss_fn, batch, self.device, optimizer, training=require_grad
        )

    def forward_interp(self, batch, optimizer=None, require_grad=True):
        if not self.interp:
            return None, 0.0
        return run_interpolator(
            self.interp, self.loss_fn, batch, self.device, optimizer, training=require_grad
        )


    def forward_synth_gan(self, batch, optimizer_synth=None, optimizer_disc=None, require_grad=True):
        if not self.synth or not self.disc:
            return None, 0.0, 0.0
        return run_synthesizer_gan(
            self.synth,
            self.disc,
            self.loss_fn,
            batch,
            self.device,
            optimizer_synth,
            optimizer_disc,
            training=require_grad,  
        )


    def log_step(self, stage, losses, batch=None, outputs=None):
        if not self.context.writer or self.global_step % 1000 != 0:
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

    def freeze(self, net):
        if not net:
            return
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

    def unfreeze(self, net):
        if not net:
            return
        net.train()
        for p in net.parameters():
            p.requires_grad = True

    def stage1_warmup(self):
        self.unfreeze(self.seg)
        self.unfreeze(self.interp)
        self.unfreeze(self.synth)
        self.unfreeze(self.disc)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            seg_out, loss_seg = self.forward_seg(
                batch, optimizer=self.optimizers["seg"]
            )
            interp_out, loss_interp = self.forward_interp(
                batch, optimizer=self.optimizers["interp"]
            )
            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch,
                optimizer_synth=self.optimizers["synth"],
                optimizer_disc=self.optimizers["disc"],
            )

            self.log_step(
                stage="Stage1",
                losses={
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
            outputs = {
                "seg": [seg.detach().cpu() for seg in seg_out] if seg_out is not None else None,
                "interp": interp_out.detach().cpu() if interp_out is not None else None,
                "synth": fake_synth_out.detach().cpu() if fake_synth_out is not None else None,
            },
            batch=batch)
            self.global_step += 1

    def stage2_frozen_seg(self):
        self.freeze(self.seg)
        self.unfreeze(self.interp)
        self.unfreeze(self.synth)
        self.unfreeze(self.disc)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            with torch.no_grad():
                seg_out, _ = self.forward_seg(batch, require_grad=False)

            self.replace_labels_with_segmentation(batch, seg_out)

            interp_out, loss_interp = self.forward_interp(
                batch, optimizer=self.optimizers["interp"]
            )

            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch,
                optimizer_synth=self.optimizers["synth"],
                optimizer_disc=self.optimizers["disc"],
            )

            self.log_step(
                stage="Stage2",
                losses={
                    "Interpolation": loss_interp,
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
                outputs={"seg": seg_out, "interp": interp_out, "synth": fake_synth_out},
                batch=batch,
            )
            self.global_step += 1

    def stage3_frozen_seg_and_interp(self):
        # Freeze teachers
        self.freeze(self.seg)
        self.freeze(self.interp)

        # Train GAN
        self.unfreeze(self.synth)
        self.unfreeze(self.disc)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            # Teacher segmentation (no grads)
            with torch.no_grad():
                seg_out, _ = self.forward_seg(batch, require_grad=False)

            self.replace_labels_with_segmentation(batch, seg_out)

            with torch.no_grad():
                interp_out, _ = self.forward_interp(batch, optimizer=None)

            self.replace_middle_with_interpolation(batch, interp_out)

            # GAN update
            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch,
                optimizer_synth=self.optimizers["synth"],
                optimizer_disc=self.optimizers["disc"],
            )

            self.log_step(
                stage="Stage3",
                losses={
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
                outputs={
                    "seg": seg_out,
                    "synth": fake_synth_out,
                },
                batch=batch,
            )
            self.global_step += 1

    def stage4_joint_finetune(self):
        self.unfreeze(self.seg)
        self.unfreeze(self.interp)
        self.unfreeze(self.synth)
        self.unfreeze(self.disc)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            for opt in ["seg", "interp", "synth", "disc"]:
                self.optimizers[opt].zero_grad(set_to_none=True)

            seg_out, loss_seg = self.forward_seg(batch, optimizer=None)
            self.replace_labels_with_segmentation(batch, seg_out)
            interp_out, loss_interp = self.forward_interp(batch, optimizer=None)
            self.replace_middle_with_interpolation(batch, interp_out)

            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch, optimizer_synth=None, optimizer_disc=None
            )

            total_loss = (
                self.opt.seg_weight * loss_seg
                + self.opt.interp_weight * loss_interp
                + self.opt.synth_weight * loss_G
            )

            total_loss.backward()

            for opt in ["seg", "interp", "synth", "disc"]:
                self.optimizers[opt].step()

            self.log_step(
                stage="Stage4",
                losses={
                    "Total": total_loss,
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
                outputs={"seg": seg_out, "interp": interp_out, "synth": fake_synth_out},
                batch=batch,
            )
            self.global_step += 1

    def _log_validation(
        self, avg_loss_seg, avg_loss_interp, avg_loss_G, avg_loss_D, avg_dice_seg, avg_dice_interp
    ):
        if self.context.logger:
            self.context.logger.info(
                    f"[Validation] Seg_Loss={avg_loss_seg:.4f}, Interp_Loss={avg_loss_interp:.4f}, Synth_G_Loss={avg_loss_G:.4f}, Synth_D_Loss={avg_loss_D:.4f}, Dice_Seg={avg_dice_seg:.4f}, Dice_Interp={avg_dice_interp:.4f}"
            )
        if self.context.writer:
            visualization.plot(
                self.context,
                {"segmentation": avg_dice_seg, "interpolation": avg_dice_interp},
                self.global_step,
                tag="dice",
            )
            visualization.plot(
                self.context,
                {"validation": avg_loss_seg},
                self.global_step,
                "loss/segmentation",
            )
            visualization.plot(
                self.context,
                {"validation": avg_loss_G},
                self.global_step,
                "loss/synthesis_G",
                )
            visualization.plot(
                self.context,
                {"validation": avg_loss_D},
                self.global_step,
                "loss/synthesis_D",
                )

            visualization.plot(
                self.context,
                {"validation": avg_loss_interp},
                self.global_step,
                "loss/interpolation",
            )

    def validate(self):
        n_batches = len(self.dataloaders.val)

        totals = {
            "loss": {
                "seg": 0.0,
                "interp": 0.0,
                "synth_G": 0.0,
                "synth_D": 0.0,
            },
            "dice": {
                "seg": 0.0,
                "interp": 0.0,
            },
            "psnr": {
                "synth": 0.0,
            },
        }

        with torch.no_grad():
            for data in self.dataloaders.val:
                batch = self.make_batch(data)

                seg_out, loss_seg = self.forward_seg(batch, require_grad=False)
                interp_out, loss_interp = self.forward_interp(batch, require_grad=False)
                fake_synth_out, loss_G, loss_D = self.forward_synth_gan(batch, require_grad=False)

                outputs = {
                    "seg": seg_out,
                    "interp": interp_out,
                    "synth": fake_synth_out,
                }

                # ---- accumulate losses ----
                totals["loss"]["seg"] += float(loss_seg)
                totals["loss"]["interp"] += float(loss_interp)
                totals["loss"]["synth_G"] += float(loss_G)
                totals["loss"]["synth_D"] += float(loss_D)

                # ---- accumulate metrics ----
                if seg_out is not None:
                    for i in range(len(seg_out)):
                        pred = seg_out[i].detach()
                        target = batch.labels[i].squeeze(1).long()
                        totals["dice"]["seg"] += dice_score_multiclass(pred, target)

                if interp_out is not None:
                    totals["dice"]["interp"] += dice_score_multiclass(
                        interp_out,
                        batch.labels[1].squeeze(1).long(),
                    )
                if fake_synth_out is not None:
                    psnr_value = psnr(
                        fake_synth_out,
                        batch.images[1].to(self.device),
                        max_val=1.0,
                    )
                    totals["psnr"]["synth"] += psnr_value.item()

                # ---- visualization ----
                if self.context.writer:
                    visualization.samples_comparison(
                        self.context,
                        batch.images,
                        batch.labels,
                        outputs,
                        self.global_step,
                        tag="val_samples",
                    )

        # ---- averages ----
        metrics = {
            "loss": {
                "seg": totals["loss"]["seg"] / n_batches,
                "interp": totals["loss"]["interp"] / n_batches,
                "synth_G": totals["loss"]["synth_G"] / n_batches,
                "synth_D": totals["loss"]["synth_D"] / n_batches,
            },
            "dice": {
                "seg": (
                    totals["dice"]["seg"] / (n_batches * 3)
                    if self.seg else 0.0
                ),
                "interp": (
                    totals["dice"]["interp"] / n_batches
                    if self.interp else 0.0
                ),
            },
            "psnr": {
                "synth": (
                    totals["psnr"]["synth"] / n_batches
                    if self.synth else 0.0
                ),
            },
        }

        # ---- logging ----
        self._log_validation(
            metrics["loss"]["seg"],
            metrics["loss"]["interp"],
            metrics["loss"]["synth_G"],
            metrics["loss"]["synth_D"],
            metrics["dice"]["seg"],
            metrics["dice"]["interp"],
        )

        return metrics

    def train(self):
        seg_val_history = deque(maxlen=5)
        interp_val_history = deque(maxlen=5)
        synth_val_history = deque(maxlen=5)
        final_val_history = deque(maxlen=10)

        for self.epoch in range(self.opt.epochs):
            if isinstance(self.dataloaders.train.sampler, DistributedSampler):
                self.dataloaders.train.sampler.set_epoch(self.epoch)

            for s in self.schedulers.values():
                s.step()

            # -------- TRAIN --------
            if self.train_stage == 0:
                self.stage1_warmup()
            elif self.train_stage == 1:
                self.stage2_frozen_seg()
            elif self.train_stage == 2:
                self.stage3_frozen_seg_and_interp()
            elif self.train_stage == 3:
                self.stage4_joint_finetune()

            # -------- VALIDATE --------
            val_metrics = self.validate()

            # -------- STAGE TRANSITIONS --------

            if self.train_stage == 0:
                seg_val_history.append(val_metrics["dice"]["seg"])
                if len(seg_val_history) == seg_val_history.maxlen:
                    ri = self.relative_improvement(seg_val_history)
                    if ri < 0.01:
                        self.train_stage = 1
                        interp_val_history.clear()
                        continue

            if self.train_stage == 1:
                interp_val_history.append(val_metrics["dice"]["interp"])
                if len(interp_val_history) == interp_val_history.maxlen:
                    ri = self.relative_improvement(interp_val_history)
                    if ri < 0.01:
                        self.train_stage = 2
                        synth_val_history.clear()
                        continue

            if self.train_stage == 2:
                synth_val_history.append(val_metrics["psnr"]["synth"])
                if len(synth_val_history) == synth_val_history.maxlen:
                    ri = self.relative_improvement(synth_val_history)
                    if ri < 0.01:
                        self.train_stage = 3
                        continue
            if self.train_stage == 3:
                final_val_history.append(val_metrics["psnr"]["synth"])
                if len(final_val_history) == final_val_history.maxlen:
                    ri = self.relative_improvement(final_val_history)
                    if ri < 0.01:
                        self.context.logger.info(
                            "Training converged. Stopping."
                        )
                        break
