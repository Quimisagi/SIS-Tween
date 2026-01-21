from torch.nn.parallel import DistributedDataParallel
import torch
from torch.utils.data import DistributedSampler

from models import Interpolator, Segmentator, Synthesizer, Discriminator
from utils import visualization
from utils.psnr import psnr
from .bundles import Batch
from .training_steps import run_segmentator, run_interpolator, run_synthesizer_gan, joint_seg_interp_synth_step
from collections import deque
from utils.dice_score import dice_score_multiclass
from utils.relative_improvement import relative_improvement
from utils.log_helper import log_step, log_validation
from spade.networks.sync_batchnorm import DataParallelWithCallback
from spade.pix2pix_model import Pix2PixModel

import os


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
            Pix2PixModel(opt).to(self.device)
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

        self.optimizers = {}
        if self.seg:
            self.optimizers["seg"] = torch.optim.Adam(
                self.seg.parameters(), lr=opt.lr_seg
            )
        if self.interp:
            self.optimizers["interp"] = torch.optim.Adam(
                self.interp.parameters(), lr=opt.lr_interp
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

        self.epoch = 0
        self.global_step = 0
        self.train_stage = 0

    def save_checkpoint(self, path: str):
        """Save the full training state to a checkpoint file."""
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "train_stage": self.train_stage,
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "opt": self.opt,  # Save config
        }

        # Save model states
        if self.seg:
            state["model"]["seg"] = self.seg.module.state_dict() if hasattr(self.seg, "module") else self.seg.state_dict()
        if self.interp:
            state["model"]["interp"] = self.interp.module.state_dict() if hasattr(self.interp, "module") else self.interp.state_dict()
        if self.synth:
            state["model"]["synth"] = self.synth.module.state_dict() if hasattr(self.synth, "module") else self.synth.state_dict()

        # Save optimizers
        for k, opt in self.optimizers.items():
            state["optimizer"][k] = opt.state_dict()

        # Save schedulers
        for k, sched in self.schedulers.items():
            state["scheduler"][k] = sched.state_dict()

        torch.save(state, path)
        if self.context.logger:
            self.context.logger.info(f"Checkpoint saved to {path}")

    def make_batch(self, data):
        """
        data: dict with keys "images" and "labels"
        """
        return Batch(images=data["images"], labels=data["labels"], edges=data["edges"], class_index=data["class_index"])


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

            log_step(
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

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            with torch.no_grad():
                seg_out, loss_seg = self.forward_seg(batch, require_grad=False)

            self.replace_labels_with_segmentation(batch, seg_out)

            interp_out, loss_interp = self.forward_interp(
                batch, optimizer=self.optimizers["interp"]
            )

            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch,
                optimizer_synth=self.optimizers["synth"],
                optimizer_disc=self.optimizers["disc"],
            )

            log_step(
                stage="Stage2",
                losses={
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
                outputs={"seg": seg_out, "interp": interp_out, "synth": fake_synth_out},
                batch=batch,
            )
            self.global_step += 1

    def stage3_frozen_seg_and_interp(self):
        self.freeze(self.seg)
        self.freeze(self.interp)

        self.unfreeze(self.synth)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            # Teacher segmentation (no grads)
            with torch.no_grad():
                seg_out, loss_seg = self.forward_seg(batch, require_grad=False)

            self.replace_labels_with_segmentation(batch, seg_out)

            with torch.no_grad():
                interp_out, loss_interp = self.forward_interp(batch, optimizer=None)

            self.replace_middle_with_interpolation(batch, interp_out)

            # GAN update
            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch,
                optimizer_synth=self.optimizers["synth"],
                optimizer_disc=self.optimizers["disc"],
            )

            log_step(
                stage="Stage3",
                losses={
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                    "Synthesis_G": loss_G,
                    "Synthesis_D": loss_D,
                },
                outputs={
                    "seg": seg_out,
                    "interp": interp_out,
                    "synth": fake_synth_out,
                },
                batch=batch,
            )
            self.global_step += 1

    def stage4_joint_finetune(self):
        self.unfreeze(self.seg)
        self.unfreeze(self.interp)
        self.unfreeze(self.synth)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            out = joint_seg_interp_synth_step(
                segmentator=self.seg,
                interpolator=self.interp,
                synthesizer_G=self.synth,
                synthesizer_D=self.disc,
                loss_fn=self.loss_fn,
                batch=batch,
                device=self.device,
                num_classes=self.opt.semantic_nc,
                optimizer_seg=self.optimizers["seg"],
                optimizer_interp=self.optimizers["interp"],
                optimizer_G=self.optimizers["synth"],
                optimizer_D=self.optimizers["disc"],
                training=True,
            )

            self.log_step(
                stage="Stage4",
                losses={
                    "Total": out["loss"],
                },
                outputs={
                    "synth": out["fake_mid"],
                },
                batch=batch,
            )

            self.global_step += 1


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
        log_validation(metrics)

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
                    ri = relative_improvement(seg_val_history)
                    if ri < 0.005:
                        self.context.logger.info(f"Segmentation converged with RI={ri:.4f}. Moving to Stage 2.")
                        self.train_stage = 1
                        interp_val_history.clear()
                        continue

            if self.train_stage == 1:
                interp_val_history.append(val_metrics["dice"]["interp"])
                if len(interp_val_history) == interp_val_history.maxlen:
                    ri = relative_improvement(interp_val_history)
                    if ri < 0.005:
                        self.context.logger.info(f"Interpolation converged with RI={ri:.4f}. Moving to Stage 3.")
                        self.train_stage = 2
                        synth_val_history.clear()
                        continue

            if self.train_stage == 2:
                synth_val_history.append(val_metrics["psnr"]["synth"])
                if len(synth_val_history) == synth_val_history.maxlen:
                    ri = relative_improvement(synth_val_history)
                    if ri < 0.005:
                        self.context.logger.info(f"Synthesis converged with RI={ri:.4f}. Moving to Stage 4.")
                        self.train_stage = 3
                        continue
            if self.train_stage == 3:
                final_val_history.append(val_metrics["psnr"]["synth"])
                if len(final_val_history) == final_val_history.maxlen:
                    ri = relative_improvement(final_val_history)
                    if ri < 0.005:
                        self.context.logger.info(
                                f"Training converged. Stopping training with RI={ri:.4f}."
                        )
                        break

            # -------- SAVE CHECKPOINT --------
            if (self.epoch + 1) % 5 == 0:
                os.makedirs(self.opt.checkpoints_dir, exist_ok=True)
                checkpoint_path = f"{self.opt.checkpoints_dir}/checkpoint_epoch_{self.epoch+1}.pth"
                self.save_checkpoint(checkpoint_path)
