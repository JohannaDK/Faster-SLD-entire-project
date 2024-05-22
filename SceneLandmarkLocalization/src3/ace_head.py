import torch
from torch import nn

class ACEHead(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ACEHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(output_channels, output_channels, kernel_size=1)

        # Skip connections using nn.Identity for demonstration, use nn.functional to add directly if preferred
        self.skip1 = nn.Identity()
        self.skip2 = nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2) + self.skip1(x1)  # Add skip connection
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5) + self.skip2(x3)  # Add another skip connection
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        return x8
