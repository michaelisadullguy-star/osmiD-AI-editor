"""
Part 2: ML Model Architecture for Feature Identification
U-Net based semantic segmentation model with contour detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is (batch, channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation

    Features detected:
    - Buildings (class 0)
    - Lawns (class 1)
    - Natural woods (class 2)
    - Artificial forests (class 3)
    - Water bodies (class 4)
    - Farmland (class 5)
    - Background (class 6)
    """

    def __init__(self, n_channels=3, n_classes=7, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ContourDetectionHead(nn.Module):
    """
    Additional head for precise contour detection
    Outputs contour probability maps for each feature class
    """

    def __init__(self, in_channels=64, n_classes=7):
        super().__init__()
        self.contour_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.contour_conv(x)


class FeatureSegmentationModel(nn.Module):
    """
    Complete model with semantic segmentation and contour detection
    """

    def __init__(self, n_channels=3, n_classes=7, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Main U-Net backbone
        self.unet = UNet(n_channels, n_classes, bilinear)

        # Contour detection head (uses features from U-Net)
        self.contour_head = ContourDetectionHead(64, n_classes)

    def forward(self, x):
        # Get intermediate features for contour detection
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        feat = self.unet.up1(x5, x4)
        feat = self.unet.up2(feat, x3)
        feat = self.unet.up3(feat, x2)
        feat = self.unet.up4(feat, x1)

        # Segmentation output
        seg_logits = self.unet.outc(feat)

        # Contour detection output
        contour_logits = self.contour_head(feat)

        return {
            'segmentation': seg_logits,
            'contours': contour_logits
        }


if __name__ == "__main__":
    # Test model
    model = FeatureSegmentationModel(n_channels=3, n_classes=7)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {output['segmentation'].shape}")
    print(f"Contour output shape: {output['contours'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
