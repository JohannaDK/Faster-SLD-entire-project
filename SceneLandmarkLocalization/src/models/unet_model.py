import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), DoubleConv(1024, 512))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), DoubleConv(512, 256))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), DoubleConv(256, 128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), DoubleConv(128, 64))
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, F.interpolate(x5, x4.shape[2:])], dim=1))
        x = self.up2(torch.cat([x3, F.interpolate(x, x3.shape[2:])], dim=1))
        x = self.up3(torch.cat([x2, F.interpolate(x, x2.shape[2:])], dim=1))
        x = self.up4(torch.cat([x1, F.interpolate(x, x1.shape[2:])], dim=1))
        logits = self.outc(x)
        return logits
