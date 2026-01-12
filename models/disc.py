import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=4,
            stride=stride,
            padding=1
        )
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    def __init__(self, image_channels=3, seg_channels=19, base_ch=64):
        super().__init__()

        in_ch = image_channels + seg_channels

        self.conv1 = nn.Conv2d(in_ch, base_ch, 4, 2, 1)
        self.conv2 = DiscriminatorBlock(base_ch, base_ch * 2)
        self.conv3 = DiscriminatorBlock(base_ch * 2, base_ch * 4)
        self.conv4 = DiscriminatorBlock(base_ch * 4, base_ch * 8, stride=1)

        self.conv_out = nn.Conv2d(base_ch * 8, 1, 4, 1, 1)

    def forward(self, img, seg):
        """
        img: (B, 3, H, W)
        seg: (B, C, H, W)  # one-hot or embedded
        """
        x = torch.cat([img, seg], dim=1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return self.conv_out(x)  # (B, 1, H', W')

