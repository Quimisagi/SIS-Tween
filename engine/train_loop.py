import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from utils.dice_score import dice_score_multiclass
from utils.visualization import samples_comparison, plot_losses
from .bundles import ContextBundle, ModelsBundle, Batch, DataloaderBundle
from .training_steps import run_segmentator, run_interpolator
from .validation import validate

def warmup_train(
    loss,
    optimizers,
    weights,
    models: ModelsBundle,
    context: ContextBundle,
    dataloaders: DataloaderBundle,
    epoch: int,
):
    models.seg.train()
    models.interp.train()

    for i, data in enumerate(dataloaders.train):
        images = data["images"]
        labels = data["labels"]

        batch = Batch(images=images, labels=labels)

        seg_output, loss_seg = run_segmentator(
            models.seg,
            loss,
            batch,
            context.device,
            optimizers["seg"],
            weights["seg"],
        )

        interp_output, loss_interp = run_interpolator(
            models.interp,
            loss,
            batch,
            context.device,
            optimizers["interp"],
            weights["interp"],
        )

        if context.writer and i % 100 == 0:
            context.logger.info(
                f"[Epoch:{epoch + 1}/{context.epochs}][Stage1][Step:{i}]: Seg_Loss={loss_seg:.4f}, Interp_Loss={loss_interp:.4f}"
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
    loss,
    optimizers,
    weights,
    models: ModelsBundle,
    context: ContextBundle,
    dataloaders: DataloaderBundle,
    epoch: int,
):
    models.seg.eval()
    for p in models.seg.parameters():
        p.requires_grad = False

    models.interp.train()

    for i, data in enumerate(dataloaders.train):
        images = data["images"]
        labels = data["labels"]

        batch = Batch(images=images, labels=labels)

        with torch.no_grad():
            seg_output, _ = run_segmentator(
                models.seg,
                loss,
                batch,
                context.device,
                optimizers["seg"],
                weights["seg"],
                training=False,
            )

        batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]
        interp_output, loss_interp = run_interpolator(
            models.interp,
            loss,
            batch,
            context.device,
            optimizers["interp"],
            weights["interp"],
        )

        if context.writer and i % 100 == 0:
            context.logger.info(
                f"[Epoch:{epoch + 1}/{context.epochs}][Stage2][Step:{i}] Interp_Loss={loss_interp:.4f}"
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

def train_loop(
    loss,
    optimizers,
    weights,
    models: ModelsBundle,
    context: ContextBundle,
    dataloaders: DataloaderBundle,
):
    train_stage = 0
    for epoch in range(context.epochs):
        if isinstance(dataloaders.train.sampler, DistributedSampler):
            dataloaders.train.sampler.set_epoch(epoch)

        if isinstance(dataloaders.val.sampler, DistributedSampler):
            dataloaders.val.sampler.set_epoch(epoch)

        if train_stage == 0:
            warmup_train(
                loss,
                optimizers,
                weights,
                models,
                context,
                dataloaders,
                epoch,
            )

        elif train_stage == 1:
            frozen_seg_train(
                loss,
                optimizers,
                weights,
                models,
                context,
                dataloaders,
                epoch,
            )

        val_loss_seg, val_loss_interp, dice_score_seg = validate(
            loss,
            optimizers,
            dataloaders.val,
            weights,
            epoch,
            models,
            context,
        )

        if train_stage == 0 and  dice_score_seg > context.segmentator_score_threshold:
            context.logger.info(
                f"Dice score {dice_score_seg:.4f} exceeded "
                f"threshold {context.segmentator_score_threshold:.4f}."
            )
            context.logger.info("Switching to Stage 2 training (frozen Segmentator).")
            train_stage = 1
