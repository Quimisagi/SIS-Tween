from torch.nn.parallel import DistributedDataParallel
import torch
from torch.utils.data import DistributedSampler

from models import Interpolator, Segmentator 
from utils import visualization
from .bundles import Batch
from .training_steps import run_segmentator, run_interpolator
from collections import deque
from utils.dice_score import dice_score_multiclass
from utils.relative_improvement import relative_improvement
from utils.log_helper import log_step, log_validation
from utils.storage import save_checkpoint, load_checkpoint

import os
import numpy as np
from PIL import Image


class Trainer:
    def __init__(self, loss_fn, dataloaders, opt, context):
        self.opt = opt
        self.context = context
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.device = context.device
        self.global_step = 0
        self.epoch = 0
        self.train_stage = 0

        self.seg = (
            Segmentator(num_seg_classes=opt.semantic_nc).to(self.device)
            if "seg" in opt.active_models else None
        )
        if self.seg and opt.distributed_enabled and context.world_size > 1:
            self.seg = DistributedDataParallel(self.seg, device_ids=[context.local_rank])

        self.interp = (
            Interpolator(sem_c=opt.semantic_nc, base_c=64).to(self.device)
            if "interp" in opt.active_models else None
        )
        if self.interp and opt.distributed_enabled and context.world_size > 1:
            self.interp = DistributedDataParallel(self.interp, device_ids=[context.local_rank])

        self.is_training = self.dataloaders.train is not None

        self.optimizers = {}
        if self.seg:
            self.optimizers["seg"] = torch.optim.Adam(self.seg.parameters(), lr=opt.lr_seg)
        if self.interp:
            self.optimizers["interp"] = torch.optim.Adam(self.interp.parameters(), lr=opt.lr_interp)

        self.schedulers = {}
        if self.seg:
            self.schedulers["seg"] = torch.optim.lr_scheduler.StepLR(self.optimizers["seg"], 10, 0.1)
        if self.interp:
            self.schedulers["interp"] = torch.optim.lr_scheduler.StepLR(self.optimizers["interp"], 10, 0.1)

    def load_checkpoint(self, path: str):
            """Load weights from a checkpoint file into models."""
            self.context.logger.info(f"Loading checkpoint: {path}")
            
            checkpoint = torch.load(path, map_location=self.device)
            
            if "model" in checkpoint:
                model_states = checkpoint["model"]
            else:
                model_states = checkpoint

            def load_weights(network, state_dict):
                try:
                    if hasattr(network, 'module'):
                        network.module.load_state_dict(state_dict)
                    else:
                        network.load_state_dict(state_dict)
                except RuntimeError as e:
                    self.context.logger.warning(f"Strict loading failed, trying strict=False. Error: {e}")
                    network.load_state_dict(state_dict, strict=False)

            if self.seg and "seg" in model_states:
                load_weights(self.seg, model_states["seg"])
                self.context.logger.info("Segmentation weights loaded.")
                
            if self.interp and "interp" in model_states:
                load_weights(self.interp, model_states["interp"])
                self.context.logger.info("Interpolation weights loaded.")


    def make_batch(self, data):
        """
        data: dict with keys "images" and "labels"
        """
        return Batch(images=data["images"], labels=data["labels"], edges=data["edges"], class_index=data["class_index"])


    def replace_labels_with_segmentation(self, batch: Batch, seg_output):
        if seg_output is not None:
            batch.labels = [
                seg.detach().argmax(dim=1, keepdim=True)
                for seg in seg_output
            ]

    def replace_middle_with_interpolation(self, batch: Batch, interp_output):
        if interp_output is not None:
            batch.labels[1] = interp_output.detach().argmax(dim=1)


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

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            seg_out, loss_seg = self.forward_seg(
                batch, optimizer=self.optimizers["seg"]
            )
            interp_out, loss_interp = self.forward_interp(
                batch, optimizer=self.optimizers["interp"]
            )

            log_step(
                context=self.context,
                opt = self.opt,
                global_step=self.global_step,
                epoch=self.epoch,
                stage="Stage1",
                losses={
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                },
            outputs = {
                "seg": [seg.detach().cpu() for seg in seg_out] if seg_out is not None else None,
                "interp": interp_out.detach().cpu() if interp_out is not None else None,
            },
            batch=batch)
            self.global_step += 1

    def stage2_frozen_seg(self):
        self.freeze(self.seg)
        self.unfreeze(self.interp)

        for data in self.dataloaders.train:
            batch = self.make_batch(data)

            # Segmentation teacher (no grad, no optimizer)
            with torch.no_grad():
                seg_out, loss_seg = self.forward_seg(batch, require_grad=False)

            self.replace_labels_with_segmentation(batch, seg_out)

            # Train interpolation
            interp_out, loss_interp = self.forward_interp(
                batch, optimizer=self.optimizers["interp"]
            )

            log_step(
                context=self.context,
                opt=self.opt,
                global_step=self.global_step,
                epoch=self.epoch,
                stage="Stage2",
                losses={
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                },
                outputs={
                    "seg": [s.detach().cpu() for s in seg_out] if seg_out is not None else None,
                    "interp": interp_out.detach().cpu() if interp_out is not None else None,
                },
                batch=batch,
            )
            self.global_step += 1

    def validate(self, save_dir=None):
        if self.seg: self.seg.eval()
        if self.interp: self.interp.eval()

        n_batches = len(self.dataloaders.val)
        totals = {"loss": {"seg": 0.0, "interp": 0.0}, "dice": {"seg": 0.0, "interp": 0.0}}

        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataloaders.val):
                batch = self.make_batch(data)
                seg_out, loss_seg = self.forward_seg(batch, require_grad=False)
                interp_out, loss_interp = self.forward_interp(batch, require_grad=False)

                totals["loss"]["seg"] += float(loss_seg)
                totals["loss"]["interp"] += float(loss_interp)

                if seg_out is not None:
                    for i in range(len(seg_out)):
                        totals["dice"]["seg"] += dice_score_multiclass(seg_out[i].detach(), batch.labels[i].squeeze(1).long())

                if interp_out is not None:
                    totals["dice"]["interp"] += dice_score_multiclass(interp_out, batch.labels[1].squeeze(1).long())

                if save_dir is not None and interp_out is not None:
                    gt_dir = os.path.join(save_dir, "gt")
                    os.makedirs(save_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)

                    preds = interp_out.detach().argmax(dim=1).cpu().numpy().astype(np.uint8)
                    
                    gt_imgs = batch.images[1]

                    for i in range(len(preds)):
                        file_name_base = f"batch{batch_idx:04d}_sample{i:02d}.png"
                        
                        Image.fromarray(preds[i]).save(os.path.join(save_dir, "pred_" + file_name_base))
                        
                        gt_img_tensor = gt_imgs[i].detach().cpu()
                        gt_img_tensor = torch.clamp((gt_img_tensor + 1.0) / 2.0, 0, 1)
                        gt_img_np = gt_img_tensor.permute(1, 2, 0).numpy()
                        
                        if gt_img_np.dtype == np.float32 or gt_img_np.dtype == np.float64:
                            gt_img_np = (np.clip(gt_img_np, 0, 1) * 255).astype(np.uint8)
                        
                        if gt_img_np.ndim == 3 and gt_img_np.shape[2] == 1:
                            gt_img_np = gt_img_np.squeeze(2)

                        Image.fromarray(gt_img_np).save(os.path.join(gt_dir, "pred_" + file_name_base))

                if self.context.writer and self.global_step % 10 == 0: 
                    visualization.samples_comparison(
                        self.context, batch.images, batch.labels, 
                        {"seg": seg_out, "interp": interp_out}, 
                        self.global_step, tag="val_samples"
                    )

        metrics = {
            "loss": {k: v / n_batches if n_batches > 0 else 0 for k, v in totals["loss"].items()},
            "dice": {
                "seg": (totals["dice"]["seg"] / (n_batches * 3) if self.seg and n_batches > 0 else 0.0),
                "interp": (totals["dice"]["interp"] / n_batches if self.interp and n_batches > 0 else 0.0),
            },
        }

        log_validation(self.context, self.global_step, metrics)
        
        if self.is_training:
            if self.seg: self.seg.train()
            if self.interp: self.interp.train()

        return metrics


    def train(self):
        seg_val_history = deque(maxlen=5)
        interp_val_history = deque(maxlen=5)

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

            # -------- VALIDATE --------
            val_metrics = self.validate()

            # -------- STAGE TRANSITIONS --------

            if self.train_stage == 0:
                seg_val_history.append(val_metrics["dice"]["seg"])
                if len(seg_val_history) == seg_val_history.maxlen:
                    ri = relative_improvement(seg_val_history)
                    if ri < 0.002:
                        self.context.logger.info(f"Segmentation converged with RI={ri:.4f}. Moving to Stage 2.")
                        self.train_stage = 1
                        interp_val_history.clear()
                        continue

            # -------- SAVE CHECKPOINT --------
            if (self.epoch + 1) % 5 == 0:
                os.makedirs(self.opt.checkpoints_dir, exist_ok=True)
                checkpoint_path = f"{self.opt.checkpoints_dir}/checkpoint_epoch_{self.epoch+1}.pth"
                save_checkpoint(path=checkpoint_path,
                                context=self.context,
                                epoch=self.epoch,
                                global_step=self.global_step,
                                train_stage = self.train_stage,
                                opt=self.opt,
                                seg=self.seg,
                                interp=self.interp,
                                optimizers=self.optimizers,
                                schedulers=self.schedulers)

    def test(self):
            self.context.logger.info("Starting Test Evaluation...")
            ckpt_path = "./checkpoints/checkpoint_epoch_30.pth"

            load_checkpoint(
                path=ckpt_path,
                context=self.context,
                device=self.device,
                seg=self.seg,
                interp=self.interp,
            )

            if not self.seg and not self.interp:
                self.context.logger.warning("No models (Seg or Interp) active!")
                return {}

            save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, "test_predictions")
            os.makedirs(save_dir, exist_ok=True)
            self.context.logger.info(f"Saving interpolation predictions to: {save_dir}")

            metrics = self.validate(save_dir=save_dir)
            
            self.context.logger.info("================ Test Results ================")
            self.context.logger.info(f"Segmentation Dice: {metrics['dice']['seg']:.4f}")
            self.context.logger.info(f"Interpolation Dice: {metrics['dice']['interp']:.4f}")
            self.context.logger.info("==============================================")
            return metrics
