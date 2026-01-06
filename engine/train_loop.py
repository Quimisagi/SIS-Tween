import torch
from torch.utils.data import DistributedSampler
from utils import visualization
from .bundles import RuntimeContext,Batch, DataloaderBundle, TrainingState
from .training_steps import run_segmentator, run_interpolator, run_synthesizer
from .validation import validate
from collections import deque
from utils import TrainConfig

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

def _make_batch(data):
    return Batch(
        images=data["images"],
        labels=data["labels"],
    )


def _forward_seg(
    training_state: TrainingState,
    batch: Batch,
    device,
    optimizer=None,
    require_grad: bool = True,
):
    if require_grad:
        return run_segmentator(
            training_state.seg,
            training_state.loss,
            batch,
            device,
            optimizer,
            training_state.weights["seg"],
        )
    else:
        with torch.no_grad():
            return run_segmentator(
                training_state.seg,
                training_state.loss,
                batch,
                device,
                optimizer=None,
                weights=training_state.weights["seg"],
            )


def _forward_interp(
    training_state: TrainingState,
    batch: Batch,
    device,
    optimizer=None,
):
    return run_interpolator(
        training_state.interp,
        training_state.loss,
        batch,
        device,
        optimizer,
        training_state.weights["interp"],
    )

def _forward_synth(
        training_state: TrainingState,
        batch: Batch,
        device,
        optimizer=None,
        ):
    return run_synthesizer(
        training_state.synth,
        training_state.loss,
        batch,
        device,
        optimizer,
        weights=training_state.weights["synth"],
    )


def _replace_labels_with_segmentation(batch: Batch, seg_output):
    batch.labels = [seg.detach().argmax(dim=1) for seg in seg_output]


def _log_step(
    context: RuntimeContext,
    stage: str,
    epoch: int,
    cfg: TrainConfig,
    global_step: int,
    losses: dict,
    images=None,
    labels=None,
    outputs=None,
):
    if not context.writer or global_step % 100 != 0:
        return

    loss_str = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
    context.logger.info(
        f"[{stage}][Epoch:{epoch + 1}/{cfg.epochs}][Step:{global_step}] {loss_str}"
    )

    if "Segmentation" in losses:
        visualization.plot(
                context,
                {"train" : losses["Segmentation"]},
            global_step,
            "loss/segmentation",
        )

    if "Interpolation" in losses:
        visualization.plot(
            context,
            {"train" : losses["Interpolation"]},
            global_step,
            "loss/interpolation",
        )

    if images is not None:
        visualization.samples_comparison(
            context,
            images,
            labels,
            outputs,
            global_step,
            tag="samples",
        )


# ============================================================
# Stage 1 — Warmup
# ============================================================

def warmup_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
):
    training_state.seg.train()
    training_state.interp.train()

    for data in dataloaders.train:
        batch = _make_batch(data)

        seg_output, loss_seg = _forward_seg(
            training_state,
            batch,
            context.device,
            optimizer=training_state.optimizers["seg"],
        )

        interp_output, loss_interp = _forward_interp(
            training_state,
            batch,
            context.device,
            optimizer=training_state.optimizers["interp"],
        )
        synth_output, loss_synth = _forward_synth(
            training_state,
            batch,
            context.device,
            optimizer=training_state.optimizers["synth"],
            )

        _log_step(
            context,
            stage="Stage1",
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
            losses={
                "Segmentation": loss_seg,
                "Interpolation": loss_interp,
                "Synthesis": loss_synth,
            },
            images=batch.images,
            labels=batch.labels,
            outputs={"seg": seg_output, "interp": interp_output, "synth": synth_output},
        )

        global_step += 1

    return global_step


# ============================================================
# Stage 2 — Frozen segmentation
# ============================================================

def frozen_seg_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
):
    training_state.seg.eval()
    training_state.interp.train()

    for p in training_state.seg.parameters():
        p.requires_grad = False

    for data in dataloaders.train:
        batch = _make_batch(data)

        seg_output, _ = _forward_seg(
            training_state,
            batch,
            context.device,
            require_grad=False,
        )

        _replace_labels_with_segmentation(batch, seg_output)

        interp_output, loss_interp = _forward_interp(
            training_state,
            batch,
            context.device,
            optimizer=training_state.optimizers["interp"],
        )

        outputs = {
            "seg": seg_output,
            "interp": interp_output,
        }

        _log_step(
            context,
            stage="Stage2",
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
            losses={"Interpolation": loss_interp},
            images=batch.images,
            labels=batch.labels,
            outputs=outputs,
        )

        global_step += 1

    return global_step


# ============================================================
# Stage 3 — Joint finetuning
# ============================================================

def joint_finetune_train(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
):
    training_state.seg.train()
    training_state.interp.train()

    for data in dataloaders.train:
        batch = _make_batch(data)

        training_state.optimizers["interp"].zero_grad()
        training_state.optimizers["seg_finetune"].zero_grad()

        seg_output, loss_seg = _forward_seg(
            training_state,
            batch,
            context.device,
            optimizer=None,
        )

        _replace_labels_with_segmentation(batch, seg_output)

        interp_output, loss_interp = _forward_interp(
            training_state,
            batch,
            context.device,
            optimizer=None,
        )

        total_loss = (
            training_state.weights["seg"] * loss_seg
            + training_state.weights["interp"] * loss_interp
        )

        total_loss.backward()
        training_state.optimizers["interp"].step()
        training_state.optimizers["seg_finetune"].step()

        _log_step(
            context,
            stage="Stage3",
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
            losses={
                "Total": total_loss,
                "Segmentation": loss_seg,
                "Interpolation": loss_interp,
            },
        )

        global_step += 1

    return global_step


def train_loop(
    training_state: TrainingState,
    context: RuntimeContext,
    dataloaders: DataloaderBundle,
    cfg: TrainConfig,
):
    train_stage = 0
    interp_val_history = deque(maxlen=5)

    global_step = 0

    for epoch in range(cfg.epochs):
        if isinstance(dataloaders.train.sampler, DistributedSampler):
            dataloaders.train.sampler.set_epoch(epoch)

        if isinstance(dataloaders.val.sampler, DistributedSampler):
            dataloaders.val.sampler.set_epoch(epoch)

        training_state.schedulers["seg"].step()
        training_state.schedulers["interp"].step()


        if train_stage == 0:
            global_step = warmup_train(
                training_state,
                context,
                dataloaders,
                cfg,
                epoch,
                global_step,
            )

        elif train_stage == 1:
            global_step = frozen_seg_train(
                training_state,
                context,
                dataloaders,
                cfg,
                epoch,
                global_step,
            )

        elif train_stage == 2:
            global_step = joint_finetune_train(
                training_state,
                context,
                dataloaders,
                cfg,
                epoch,
                global_step,
            )

        val_loss_seg, val_loss_interp, dice_score_seg = validate(
            training_state,
            context,
            dataloaders.val,
            global_step,
        )

        if train_stage == 0 and dice_score_seg > cfg.segmentator_score_threshold:
            context.logger.info(
                f"Dice score {dice_score_seg:.4f} exceeded "
                f"threshold {cfg.segmentator_score_threshold:.4f}. "
                "Switching to Stage 2."
            )
            train_stage = 1

        if train_stage == 1:
            interp_val_history.append(val_loss_interp)

            if len(interp_val_history) == interp_val_history.maxlen:
                ri = relative_improvement_window(list(interp_val_history))

                context.logger.info(
                    f"[Interp] Relative improvement: {ri:.4%}"
                )

                if ri < 0.01:
                    context.logger.info(
                        "Interpolation plateaued. Switching to Stage 3."
                    )
                    train_stage = 2
