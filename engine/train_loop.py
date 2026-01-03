import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from utils.dice_score import dice_score_multiclass
from utils.visualization import samples_comparison, plot_losses
from .bundles import ContextBundle, ModelsBundle, Batch, DataloaderBundle
from .training_steps import run_segmentator, run_interpolator
from .validation import validate

def train_loop(loss, optimizers, weights, models : ModelsBundle,  context : ContextBundle, dataloaders : DataloaderBundle):

    for epoch in range(context.epochs):
        if isinstance(dataloaders.train.sampler, DistributedSampler):
            dataloaders.train.sampler.set_epoch(epoch)

        models.seg.train()
        models.interp.train()

        for i, data in enumerate(dataloaders.train):
            images = data['images']
            labels = data['labels']

            batch = Batch(images=images, labels=labels)

            seg_output, loss_seg = run_segmentator(models.seg, loss, batch, context.device, optimizers["seg"], weights["seg"])
            interp_output, loss_interp = run_interpolator(models.interp, loss, batch, context.device, optimizers["interp"], weights["interp"])

            if context.writer and i % 100 == 0:
                context.logger.info(f"[Step:{i}]: Seg_Loss={loss_seg:.4f}, Interp_Loss={loss_interp:.4f}")
                plot_losses(context.writer, context.logger, {'Segmentation': loss_seg, 'Interpolation': loss_interp}, epoch * len(dataloaders.train) + i)
                samples_comparison(context.writer, context.logger, images, labels, seg_output, interp_output, epoch, tag="train_samples")

        val_loss_seg, val_loss_interp, dice_score_seg = validate(loss, optimizers, dataloaders.val, weights, epoch, models, context)
        context.logger.info(f"[Epoch:{epoch + 1}/{context.epochs}] Validation Seg_Loss={val_loss_seg:.4f}, Interp_Loss={val_loss_interp:.4f}, Dice_Score={dice_score_seg:.4f}")

        if dice_score_seg > context.segmentator_score_threshold:
            context.logger.info(f"Dice score {dice_score_seg:.4f} exceeded threshold {context.segmentator_score_threshold:.4f}.")

