import torch

PALETTE = torch.tensor(
    [
        [1, 1, 1],  # BACKGROUND
        [255, 0, 255],  # HAIR
        [255, 246, 1],  # FACE
        [255, 255, 255],  # EYE
        # [255, 1,   1],     # MOUTH
        [255, 201, 99],  # SKIN
        [0, 255, 0],  # CLOTHES
    ],
    dtype=torch.uint8,
)  # (6, 3)


def seg_to_rgb(seg: torch.Tensor) -> torch.Tensor:
    B, C, H, W = seg.shape
    assert (
        C == PALETTE.shape[0]
    ), f"Expected {PALETTE.shape[0]} channels for segmentation input, got {C}"

    # Flatten segmentation to (B, H*W, C)
    seg_flat = seg.permute(0, 2, 3, 1).reshape(B, -1, C)

    # Compute class index per pixel
    cls = seg_flat.argmax(dim=-1)  # (B, H*W)

    # Map class index → RGB
    rgb_flat = PALETTE.to(seg.device)[cls]  # (B, H*W, 3)

    # Reshape back to image format (B, 3, H, W)
    rgb = rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

    # Normalize to [-1, 1]
    rgb = rgb.float() / 127.5 - 1.0

    return rgb
