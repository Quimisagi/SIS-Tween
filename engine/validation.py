import torch
from .training_steps import run_segmentator, run_interpolator
from utils.dice_score import dice_score_multiclass
from utils import visualization
from .bundles import RuntimeContext, Batch, TrainingState


def validate(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloader,
    global_step
        ):
    training_state.seg.eval()
    training_state.interp.eval()
    total_loss_seg = 0.0
    total_loss_interp = 0.0
    total_dice_seg = 0.0
    total_dice_interp = 0.0

    n_batches = len(dataloader)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data['images']
            labels = data['labels']

            batch = Batch(images=images, labels=labels)

            seg_output, loss_seg = run_segmentator(
                training_state.seg, training_state.loss, batch, context.device, training_state.optimizers["seg"], training_state.weights["seg"], training=False)
            interp_output, loss_interp = run_interpolator(training_state.interp, training_state.loss, batch, context.device, training_state.optimizers["interp"], training_state.weights["interp"], training=False)

            total_loss_seg += loss_seg
            total_loss_interp += loss_interp

            for i in range(len(seg_output)):
                seg_output[i] = seg_output[i].detach()
                labels[i] = labels[i].to(context.device).squeeze(1).long()
                dice = dice_score_multiclass(seg_output[i], labels[i])
                total_dice_seg += dice

            dice_interp = dice_score_multiclass(interp_output, labels[1])
            total_dice_interp += dice_interp

            visualization.samples_comparison(context, images, labels, seg_output, interp_output, global_step, tag="val_samples")

    avg_loss_seg = total_loss_seg / n_batches
    avg_loss_interp = total_loss_interp / n_batches
    avg_dice_seg = total_dice_seg / n_batches / 3
    avg_dice_interp = total_dice_interp / n_batches

    if context.logger:
        context.logger.info(f"[Validation] Seg_Loss={avg_loss_seg:.4f}, Interp_Loss={avg_loss_interp:.4f}, Seg_Dice={avg_dice_seg:.4f}, Interp_Dice={avg_dice_interp:.4f}")

    if context.writer:
        dice_scalars = {
                "segmentation_loss": avg_dice_seg,
                "interpolation_loss": avg_dice_interp,
        }
        visualization.plot(context,dice_scalars, global_step, tag="dice")
        visualization.plot(context, {"validation": avg_loss_seg}, global_step, "loss/segmentation")
    visualization.plot(context, {"validation": avg_loss_interp}, global_step, "loss/interpolation")

    return avg_loss_seg, avg_loss_interp, avg_dice_seg
