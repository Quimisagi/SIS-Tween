import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from utils.dice_score import dice_score_multiclass
from utils.visualization import samples_comparison, plot_losses
from .bundles import RuntimeContext,Batch, DataloaderBundle, TrainingState
from .training_steps import run_segmentator, run_interpolator
from .validation import validate
from collections import deque

def relative_improvement_window(
    values: list[float],
    eps: float = 1e-8,
) -> float:
    """
    Computes relative improvement over a window.
    Positive -> improvement
    Near zero -> plateau
    Negative -> regression
    """
    if len(values) < 2:
        return 0.0

    old = values[0]
    new = values[-1]

    return (old - new) / max(abs(old), eps)

def warmup_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    epoch: int,
):
    training_state.seg.train()
    training_state.interp.train()

    for i, data in enumerate(dataloaders.train):
        images = data["images"]
        labels = data["labels"]

        batch = Batch(images=images, labels=labels)

        seg_output, loss_seg = run_segmentator(
            training_state.seg,
            training_state.loss,
            batch,
            context.device,
            training_state.optimizers["seg"],
            training_state.weights["seg"],
        )

        interp_output, loss_interp = run_interpolator(
            training_state.interp,
            training_state.loss,
            batch,
            context.device,
            training_state.optimizers["interp"],
            training_state.weights["interp"],
        )

        if context.writer and i % 100 == 0:
            context.logger.info(
                f"[Stage1][Epoch:{epoch + 1}/{context.epochs}][Step:{i}]: Seg_Loss={loss_seg:.4f}, Interp_Loss={loss_interp:.4f}"
            )

            plot_losses(
                context.writer,
                context.logger,
                {
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                },
                epoch * len(dataloaders.train) + i,
            )

            samples_comparison(
                context.writer,
                context.logger,
                images,
                labels,
                seg_output,
                interp_output,
                epoch,
                tag="train1_train",
            )

def frozen_seg_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    epoch: int,
):
    training_state.seg.eval()
    for p in training_state.seg.parameters():
        p.requires_grad = False

    training_state.interp.train()

    for i, data in enumerate(dataloaders.train):
        images = data["images"]
        labels = data["labels"]

        batch = Batch(images=images, labels=labels)

        with torch.no_grad():
            seg_output, _ = run_segmentator(
                training_state.seg,
                training_state.loss,
                batch,
                context.device,
                training_state.optimizers["seg"],
                training_state.weights["seg"],
                training=False,
            )

        batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]
        interp_output, loss_interp = run_interpolator(
            training_state.interp,
            training_state.loss,
            batch,
            context.device,
            training_state.optimizers["interp"],
            training_state.weights["interp"],
        )

        if context.writer and i % 100 == 0:
            context.logger.info(
                f"[Stage2][Epoch:{epoch + 1}/{context.epochs}][Step:{i}] Interp_Loss={loss_interp:.4f}"
            )

            plot_losses(
                context.writer,
                context.logger,
                {"Interpolation": loss_interp},
                epoch * len(dataloaders.train) + i,
            )

            samples_comparison(
                context.writer,
                context.logger,
                images,
                labels,
                seg_output,
                interp_output,
                epoch,
                tag="stage2_train",
            )

def joint_finetune_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    epoch: int,
):
    training_state.seg.train()
    training_state.interp.train()

    for i, data in enumerate(dataloaders.train):
        images = data["images"]
        labels = data["labels"]

        batch = Batch(images=images, labels=labels)

        # ---- zero grads ----
        training_state.optimizers["interp"].zero_grad()
        training_state.optimizers["seg_finetune"].zero_grad()

        # ---- forward seg (grad ON) ----
        seg_output, loss_seg = run_segmentator(
            training_state.seg,
            training_state.loss,
            batch,
            context.device,
            optimizer=None,   
            weights=training_state.weights["seg"],
        )

        batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]

        # ---- forward interp ----
        interp_output, loss_interp = run_interpolator(
            training_state.interp,
            training_state.loss,
            batch,
            context.device,
            optimizer=None,
            weights=training_state.weights["interp"],
        )

        # ---- combined loss (weighted) ----
        total_loss = (
            training_state.weights["seg"] * loss_seg +
            training_state.weights["interp"] * loss_interp
        )

        total_loss.backward()

        # ---- step ----
        training_state.optimizers["interp"].step()
        training_state.optimizers["seg_finetune"].step()

        # ---- logging ----
        if context.writer and i % 100 == 0:
            context.logger.info(
                f"[Stage3][Step:{i}] "
                f"Total={total_loss:.4f} "
                f"Seg={loss_seg:.4f} "
                f"Interp={loss_interp:.4f}"
            )

            plot_losses(
                context.writer,
                context.logger,
                {
                    "Total": total_loss,
                    "Segmentation": loss_seg,
                    "Interpolation": loss_interp,
                },
                epoch * len(dataloaders.train) + i,
            )


def train_loop(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
):
    train_stage = 0
    interp_val_history = deque(maxlen=5)
    for epoch in range(context.epochs):
        if isinstance(dataloaders.train.sampler, DistributedSampler):
            dataloaders.train.sampler.set_epoch(epoch)

        if isinstance(dataloaders.val.sampler, DistributedSampler):
            dataloaders.val.sampler.set_epoch(epoch)

        if train_stage == 0:
            warmup_train(
                training_state,
                context,
                dataloaders,
                epoch,
            )

        elif train_stage == 1:
            frozen_seg_train(
                training_state,
                context,
                dataloaders,
                epoch,
            )

        elif train_stage == 2:
            joint_finetune_train(
                training_state,
                context,
                dataloaders,
                epoch,
            )

        val_loss_seg, val_loss_interp, dice_score_seg = validate(
            training_state,
            context,
            dataloaders.val,
            epoch,
        )

        if train_stage == 0 and  dice_score_seg > context.segmentator_score_threshold:
            context.logger.info(
                f"Dice score {dice_score_seg:.4f} exceeded "
                f"threshold {context.segmentator_score_threshold:.4f}."
            )
            context.logger.info("Switching to Stage 2 training (frozen Segmentator).")
            train_stage = 1

        if train_stage == 1: 
            interp_val_history.append(val_loss_interp)

            interp_plateau = False
            if len(interp_val_history) == interp_val_history.maxlen:
                ri = relative_improvement_window(list(interp_val_history))

                context.logger.info(
                    f"[Interp] Relative improvement over last "
                    f"{len(interp_val_history)} epochs: {ri:.4%}"
                )

                interp_plateau = ri < 0.01  # 1% threshold

            if interp_plateau:
                context.logger.info(
                    "Interpolation validation loss plateaued. "
                    "Switching to Stage 3 training (joint finetune)."
                )
                train_stage = 2

