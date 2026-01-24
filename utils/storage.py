import torch

def load_checkpoint(context, path: str, device, seg=None, interp=None):
    """
    Load weights from a checkpoint file into models.
    """
    context.logger.info(f"Loading checkpoint: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    if "model" in checkpoint:
        model_states = checkpoint["model"]
    else:
        model_states = checkpoint

    def load_weights(network, state_dict):
        try:
            # Handle DDP 'module.' prefix mismatches
            if hasattr(network, 'module'):
                network.module.load_state_dict(state_dict)
            else:
                network.load_state_dict(state_dict)
        except RuntimeError as e:
            context.logger.warning(f"Strict loading failed, trying strict=False. Error: {e}")
            network.load_state_dict(state_dict, strict=False)

    if seg and "seg" in model_states:
        load_weights(seg, model_states["seg"])
        context.logger.info("Segmentation weights loaded.")
        
    if interp and "interp" in model_states:
        load_weights(interp, model_states["interp"])
        context.logger.info("Interpolation weights loaded.")


def save_checkpoint(path: str, context, epoch, global_step, train_stage, opt, seg=None, interp=None, optimizers=None, schedulers=None):
    """
    Save the full training state to a checkpoint file.
    """
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "train_stage": train_stage,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "opt": opt, 
    }

    if seg:
        state["model"]["seg"] = seg.module.state_dict() if hasattr(seg, "module") else seg.state_dict()
    if interp:
        state["model"]["interp"] = interp.module.state_dict() if hasattr(interp, "module") else interp.state_dict()

    if optimizers:
        for k, opt_item in optimizers.items():
            state["optimizer"][k] = opt_item.state_dict()

    if schedulers:
        for k, sched in schedulers.items():
            state["scheduler"][k] = sched.state_dict()

    torch.save(state, path)
    
    if context.logger:
        context.logger.info(f"Checkpoint saved to {path}")
