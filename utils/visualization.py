from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F
import torch 
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from . import seg_to_rgb


def init_tensorboard(cfg, local_rank):
    writer = SummaryWriter(log_dir=cfg.tensorboard_logs_dir) if (
            not cfg.distributed_enabled or local_rank == 0
        ) else None
    return writer

def _to_3ch(x):
    """
    Convert tensor to 3-channel format.
    Supports HW, CHW, and BCHW.
    """
    if x.dim() == 2:              # HW
        x = x.unsqueeze(0)        # 1HW

    if x.dim() == 3:              # CHW
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)

    elif x.dim() == 4:            # BCHW
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

    return x

def handle_visualization_labels(label):
    """ Convert labels to 3-channel RGB format for visualization. """
    label_one_hot = F.one_hot(
        label.squeeze(1).long(),
        num_classes=6
    ).permute(0, 3, 1, 2).float()
    label_rgb = seg_to_rgb.seg_to_rgb(label_one_hot)
    label_rgb = _to_3ch(label_rgb).cpu()
    return label_rgb

def _add_label_to_tensor(tensor, label_text, font_size=24):
    """Helper to add text labels to a single (C, H, W) tensor."""
    img = TF.to_pil_image(tensor.cpu())
    
    # Draw text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    x, y = 10, 10
    draw.text((x+1, y+1), label_text, fill="black", font=font)
    draw.text((x, y), label_text, fill="white", font=font)
    
    return TF.to_tensor(img)

def _denormalize(tensor):
    """Brings a [-1, 1] tensor to [0, 1] for visualization."""
    return torch.clamp((tensor + 1.0) / 2.0, 0, 1)

def samples_comparison(
    context,
    gt_images,
    gt_labels,
    seg_labels,
    interp_label,
    global_step,
    tag="samples",
):
    writer = context.writer
    logger = context.logger

    if writer is None:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    try:
        def preprocess(img_tensor, is_input_img=False, size=(256, 256)):
            img = _to_3ch(img_tensor.detach().cpu())[0]
            if is_input_img:
                img = _denormalize(img)
            return TF.resize(img, size, interpolation=TF.InterpolationMode.NEAREST)

        row_gt_img = [preprocess(img, True) for img in gt_images]
        row_gt_lbl = [preprocess(handle_visualization_labels(lbl)) for lbl in gt_labels]
        row_seg    = [preprocess(seg_to_rgb.seg_to_rgb(lbl.detach())) for lbl in seg_labels]

        interp_img = preprocess(seg_to_rgb.seg_to_rgb(interp_label.detach()))
        empty      = torch.zeros_like(interp_img)
        row_interp = [empty, interp_img, empty]

        all_images = row_gt_img + row_gt_lbl + row_seg + row_interp
        text_labels = [
            "GT_IMG_0", "GT_IMG_1", "GT_IMG_2",
            "GT_LBL_0", "GT_LBL_1", "GT_LBL_2",
            "SEG_0",    "SEG_1",    "SEG_2",
            "",         "INTERP",   ""
        ]

        labeled_images = []
        for img, txt in zip(all_images, text_labels):
            if txt != "":
                labeled_images.append(_add_label_to_tensor(img, txt))
            else:
                labeled_images.append(img)

        grid = make_grid(labeled_images, nrow=3, padding=4)
        writer.add_image(tag, grid, global_step)

        for i, (img, txt) in enumerate(zip(all_images, text_labels)):
            img_labeled = _add_label_to_tensor(img, txt) if txt != "" else img
            writer.add_image(f"{tag}/{txt or f'img_{i}'}", img_labeled, global_step)

    except Exception as e:
        logger.error(f"Visualization failed: {e}")

def plot(context, values, global_step, tag=None):
    """
    Flexible TensorBoard plotting helper.

    Supported:
    - plot(writer, logger, scalar, step, "tag")
    - plot(writer, logger, {"loss": scalar}, step, "tag")
    - plot(writer, logger, {"loss": x, "dice": y}, step, "tag")
    - plot(writer, logger, {"tag1": {...}, "tag2": {...}}, step)
    """

    if context.writer is None:
        return

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    try:
        # Case 1: single scalar
        if isinstance(values, (int, float)):
            if tag is None:
                raise ValueError("tag must be provided for scalar values")
            context.writer.add_scalar(tag, values, global_step)
            return

        # Case 2: dict of scalars -> single plot
        if tag is not None:
            context.writer.add_scalars(tag, values, global_step)
            return

        # Case 3: dict of dicts -> multiple plots
        for plot_tag, scalars in values.items():
            context.writer.add_scalars(plot_tag, scalars, global_step)

    except Exception as e:
        context.logger.error(f"Logging failed: {e}")
