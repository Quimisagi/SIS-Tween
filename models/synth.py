"""
Minimal SPADE (GauGAN) Generator
VAE-latent ONLY version

Adapted from NVIDIA GauGAN (2019)
CC BY-NC-SA 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================
# Config
# ============================================================

@dataclass
class SPADEConfig:
    semantic_nc: int 
    ngf: int = 64
    z_dim: int = 256
    crop_size: int = 256
    aspect_ratio: float = 1.0
    num_upsampling_layers: str = "normal"  # normal | more | most


# ============================================================
# SPADE Normalization
# ============================================================

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        hidden_nc = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        x = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return x * (1 + gamma) + beta


# ============================================================
# SPADE ResNet Block
# ============================================================

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, config: SPADEConfig):
        super().__init__()

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.norm_0 = SPADE(fin, config.semantic_nc)
        self.conv_0 = nn.Conv2d(fin, fmiddle, 3, padding=1)

        self.norm_1 = SPADE(fmiddle, config.semantic_nc)
        self.conv_1 = nn.Conv2d(fmiddle, fout, 3, padding=1)

        if self.learned_shortcut:
            self.norm_s = SPADE(fin, config.semantic_nc)
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2))

        return x_s + dx

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x = self.conv_s(self.norm_s(x, seg))
        return x


# ============================================================
# SPADE Generator
# ============================================================

class SPADEGenerator(nn.Module):
    def __init__(self, config: SPADEConfig):
        super().__init__()
        self.config = config
        nf = config.ngf

        self.sw, self.sh = self._compute_latent_vector_size()

        # --- VAE latent projection (MANDATORY) ---
        self.fc = nn.Linear(
            config.z_dim, 16 * nf * self.sw * self.sh
        )

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, config)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, config)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, config)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, config)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, config)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, config)
        self.up_3 = SPADEResnetBlock(2 * nf, nf, config)

        final_nc = nf
        if config.num_upsampling_layers == "most":
            self.up_4 = SPADEResnetBlock(nf, nf // 2, config)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    # --------------------------------------------------------

    def _compute_latent_vector_size(self):
        if self.config.num_upsampling_layers == "normal":
            num_up = 5
        elif self.config.num_upsampling_layers == "more":
            num_up = 6
        elif self.config.num_upsampling_layers == "most":
            num_up = 7
        else:
            raise ValueError("Invalid num_upsampling_layers")

        sw = self.config.crop_size // (2 ** num_up)
        sh = round(sw / self.config.aspect_ratio)
        return sw, sh

    # --------------------------------------------------------

    def forward(self, segmap, latent_z):
        """
        segmap  : [B, semantic_nc, H, W]  (one-hot)
        latent_z: [B, z_dim]              (from pretrained VAE)
        """

        if latent_z is None:
            raise ValueError("latent_z is required (VAE-only generator)")

        x = self.fc(latent_z)
        x = x.view(
            -1, 16 * self.config.ngf, self.sh, self.sw
        )

        x = self.head_0(x, segmap)

        x = self.up(x)
        x = self.G_middle_0(x, segmap)

        if self.config.num_upsampling_layers in ("more", "most"):
            x = self.up(x)

        x = self.G_middle_1(x, segmap)

        x = self.up(x)
        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)

        if self.config.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, segmap)

        x = torch.tanh(self.conv_img(F.leaky_relu(x, 0.2)))
        return x

class Synthesizer(nn.Module):
    def __init__(self, vae, image_size=128):
        super().__init__()
        self.vae = vae

        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        # ---- Diffusers VAE latent size ----
        self.latent_channels = vae.config.latent_channels  # usually 4
        self.latent_hw = image_size // 8                    # usually 32
        self.vae_z_dim = self.latent_channels * self.latent_hw * self.latent_hw

        self.spade_z_dim = 128
        self.z_proj = nn.Linear(self.vae_z_dim * 2, self.spade_z_dim)

        spade_config = SPADEConfig(
            semantic_nc=6,
            ngf=64,
            z_dim=self.spade_z_dim,
            crop_size=image_size,
            aspect_ratio=1.0,
            num_upsampling_layers="normal",
        )

        self.generator = SPADEGenerator(spade_config)

    def encode(self, images):
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            z = latent_dist.sample()    # [B, 4, 32, 32]
            return z

    def forward(self, frame1, frame2, segmap):
        z1 = self.encode(frame1)
        z2 = self.encode(frame2)

        z = torch.cat([z1, z2], dim=1)
        z = z.flatten(start_dim=1)
        z = self.z_proj(z)

        return self.generator(segmap, z)
