from torch.nn.parallel import DistributedDataParallel
from diffusers import AutoencoderKL
import torch
from torch.utils.data import DistributedSampler

from models import Interpolator, Segmentator, Synthesizer, Discriminator
from utils import visualization
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
                    self.seg, device_ids=[opt.local_rank]
                )
            if self.interp:
                self.interp = DistributedDataParallel(
                    self.interp, device_ids=[opt.local_rank]
                )
            if self.synth:
                self.synth = DistributedDataParallel(
                    self.synth, device_ids=[opt.local_rank]
                )
            if self.disc:
                self.disc = DistributedDataParallel(
                    self.disc, device_ids=[opt.local_rank]
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
        Computes the relative improvement between the first and last values in a list.
        Used to determine when to switch training stages.
        values: list of float
        returns: (old - new) / max(|old|, eps)
        """
        if len(values) < 2:
            return 0.0
        old = values[0]
        new = values[-1]
        return (old - new) / max(abs(old), eps)

    def replace_labels_with_segmentation(self, batch: Batch, seg_output):
        if seg_output is not None:
            batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]

    def forward_seg(self, batch, optimizer=None, require_grad=True):
        if not self.seg:
            return None, 0.0
        return run_segmentator(
            self.seg, self.loss_fn, batch, self.device, optimizer, training=require_grad
        )

    def forward_interp(self, batch, optimizer=None):
        if not self.interp:
            return None, 0.0
        return run_interpolator(
            self.interp, self.loss_fn, batch, self.device, optimizer
        )

    def forward_synth(self, batch, optimizer=None):
        if not self.synth:
            return None, 0.0
        return run_synthesizer(self.synth, self.loss_fn, batch, self.device, optimizer)

    def forward_synth_gan(self, batch, optimizer_synth=None, optimizer_disc=None):
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
                outputs={"seg": seg_out, "interp": interp_out, "synth": fake_synth_out},
            )
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
                self.optimizers[opt].zero_grad()

            seg_out, loss_seg = self.forward_seg(batch, optimizer=None)
            self.replace_labels_with_segmentation(batch, seg_out)
            interp_out, loss_interp = self.forward_interp(batch, optimizer=None)

            fake_synth_out, loss_G, loss_D = self.forward_synth_gan(
                batch, optimizer_synth=None, optimizer_disc=None
            )

            total_loss = (
                self.opt.seg_weight * loss_seg
                + self.opt.interp_weight * loss_interp
                + self.opt.g_weight * loss_G
                + self.opt.d_weight * loss_D
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
        total_loss_seg = total_loss_interp = total_loss_G = total_loss_D = total_dice_seg = total_dice_interp = 0.0
        n_batches = len(self.dataloaders.val)
        with torch.no_grad():
            for data in self.dataloaders.val:
                batch = self.make_batch(data)
                seg_out, loss_seg = self.forward_seg(batch)
                interp_out, loss_interp = self.forward_interp(batch)
                fake_synth_out, loss_G, loss_D = self.forward_synth_gan(batch)
                outputs = {"seg": seg_out, "interp": interp_out, "synth": fake_synth_out}
                total_loss_seg += loss_seg
                total_loss_interp += loss_interp
                total_loss_G += loss_G
                total_loss_D += loss_D
                if seg_out is not None:
                    for i in range(len(seg_out)):
                        pred = seg_out[i].detach()
                        target = batch.labels[i].squeeze(1).long()
                        total_dice_seg += dice_score_multiclass(pred, target)
                if interp_out is not None:
                    total_dice_interp += dice_score_multiclass(
                        interp_out, batch.labels[1].squeeze(1).long()
                    )
                if self.context.writer:
                    visualization.samples_comparison(
                        self.context,
                        batch.images,
                        batch.labels,
                        outputs,
                        self.global_step,
                        tag="val_samples",
                    )
        avg_loss_seg = total_loss_seg / n_batches
        avg_loss_interp = total_loss_interp / n_batches
        avg_loss_synth_G = total_loss_G / n_batches
        avg_loss_synth_D = total_loss_D / n_batches
        avg_dice_seg = total_dice_seg / (n_batches * 3) if self.seg else 0.0
        avg_dice_interp = total_dice_interp / n_batches if self.interp else 0.0
        self._log_validation(
            avg_loss_seg, avg_loss_interp, avg_loss_synth_G, avg_loss_synth_D, avg_dice_seg, avg_dice_interp
        )
        return avg_loss_seg, avg_loss_interp, avg_loss_synth_G, avg_loss_synth_D, avg_dice_seg

    def train(self):
        seg_val_history = deque(maxlen=5)
        interp_val_history = deque(maxlen=5)
        synth_val_history = deque(maxlen=5)

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
            val_loss_seg, val_loss_interp, val_loss_synth_G, val_loss_synth_D, dice = self.validate()

            # -------- STAGE TRANSITIONS --------

            if self.train_stage == 0:
                seg_val_history.append(val_loss_seg)
                if len(seg_val_history) == seg_val_history.maxlen:
                    ri = self.relative_improvement(seg_val_history)
                    if ri < 0.01:
                        self.train_stage = 1
                        interp_val_history.clear()
                        continue

            if self.train_stage == 1:
                interp_val_history.append(val_loss_interp)
                if len(interp_val_history) == interp_val_history.maxlen:
                    ri = self.relative_improvement(interp_val_history)
                    if ri < 0.01:
                        self.train_stage = 2
                        synth_val_history.clear()
                        continue

            if self.train_stage == 2:
                synth_val_history.append(val_loss_synth_G)
                if len(synth_val_history) == synth_val_history.maxlen:
                    ri = self.relative_improvement(synth_val_history)
                    if ri < 0.01:
                        self.train_stage = 3
                        continue
