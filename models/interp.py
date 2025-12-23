"""
Interpolator
U-Net style architecture for image interpolation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, kernel_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        x = torch.cat([up, x2], dim=1)
        x = self.conv(x)
        return x

class Interpolator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.enc1 = DownSampleBlock(in_channels, 64)
        self.enc2 = DownSampleBlock(64, 128)
        self.enc3 = DownSampleBlock(128, 256)
        self.enc4 = DownSampleBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec4 = UpSampleBlock(1024, 512)
        self.dec3 = UpSampleBlock(512, 256)
        self.dec2 = UpSampleBlock(256, 128)
        self.dec1 = UpSampleBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        d1, p1 = self.enc1(x)
        d2, p2 = self.enc2(p1)
        d3, p3 = self.enc3(p2)
        d4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        u4 = self.dec4(b, d4)
        u3 = self.dec3(u4, d3)
        u2 = self.dec2(u3, d2)
        u1 = self.dec1(u2, d1)

        out = self.final_conv(u1)
        out = self.tanh(out)

        return out
