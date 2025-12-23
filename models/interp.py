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

class CrossFrameAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.query_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, frame1_features, frame2_features):
        batch_size, channels, height, width = frame1_features.size()
        
        query = self.query_conv(frame1_features).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(frame2_features).view(batch_size, -1, height * width)
        value = self.value_conv(frame2_features).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=2)
        
        attended = torch.bmm(value, attention.permute(0, 2, 1))  
        attended = attended.view(batch_size, channels, height, width)
        
        output = self.gamma * self.out_conv(attended) + frame1_features
        
        return output


class Interpolator(nn.Module):
    """
    Contains 3 downsampling blocks, a cross-frame attention mechanism, a bottleneck, and 3 upsampling blocks.
    For downsampling, each frame is processed separately, but for upsampling, skip connections from both frames 
    are concatenated.
    """
    def __init__(self, frame_c, base_c):
        super().__init__()
        self.down1 = DownSampleBlock(frame_c, base_c)
        self.down2 = DownSampleBlock(base_c, base_c * 2)
        self.down3 = DownSampleBlock(base_c * 2, base_c * 4)

        self.attention = CrossFrameAttention(base_c * 4)

        self.bottleneck = ConvBlock(base_c * 4 * 2, base_c * 8) 

        self.up3 = UpSampleBlock(base_c * 16, base_c * 4)
        self.up2 = UpSampleBlock(base_c * 8, base_c * 2)
        self.up1 = UpSampleBlock(base_c * 4, base_c)
        self.final_conv = nn.Conv2d(base_c, frame_c, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, frame1, frame2):
        f1_d1, f1_p1 = self.down1(frame1)
        f1_d2, f1_p2 = self.down2(f1_p1)
        f1_d3, f1_p3 = self.down3(f1_p2)

        f2_d1, f2_p1 = self.down1(frame2)
        f2_d2, f2_p2 = self.down2(f2_p1)
        f2_d3, f2_p3 = self.down3(f2_p2)

        f1_att = self.attention(f1_p3, f2_p3)
        f2_att = self.attention(f2_p3, f1_p3)

        bottleneck_input = torch.cat([f1_att, f2_att], dim=1)
        bottleneck = self.bottleneck(bottleneck_input)

        skip3 = torch.cat([f1_d3, f2_d3], dim=1) 
        up3 = self.up3(bottleneck, skip3)

        skip2 = torch.cat([f1_d2, f2_d2], dim=1)
        up2 = self.up2(up3, skip2)

        skip1 = torch.cat([f1_d1, f2_d1], dim=1)
        up1 = self.up1(up2, skip1)

        out = self.final_conv(up1)
        out = self.tanh(out)

        return out

