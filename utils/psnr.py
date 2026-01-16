import torch
import torch.nn as nn

def psnr(pred, target, max_val=1.0):
    mse = nn.functional.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))
