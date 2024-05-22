import torch
import torch.nn as nn

from .blocks import _make_encoder


class ASPP(nn.Module):
    def __init__(self, in_ch, d1, d2, d3, d4, reduction=4):
        super(ASPP, self).__init__()
        self.aspp_d1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d1, dilation=d1),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )
        self.aspp_d2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d2, dilation=d2),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d3, dilation=d3),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )

        self.aspp_d4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d4, dilation=d4),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.aspp_d1(x)
        d2 = self.aspp_d2(x)
        d3 = self.aspp_d3(x)
        d4 = self.aspp_d4(x)
        return torch.cat((d1, d2, d3, d4), dim=1)


class BackboneV1(torch.nn.Module):
    """
    Backbone network v1
    """

    def __init__(self, model, path=None, output_downsample=8, features=320):

        super(BackboneV1, self).__init__()

        self.pretrained, channels = _make_encoder(use_pretrained=True, model = model, output_downsample=output_downsample)

        self.output_channels = channels

        if path:
            self.load(path)

    def forward(self, x):

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        return layer_4


class SceneHeadV1(torch.nn.Module):

    """
    Scene head network v1
    """

    def __init__(self, path=None, num_landmarks=200, features=320):

        super(SceneHeadV1, self).__init__()

        self.aspp = nn.Sequential(
            ASPP(in_ch=features, d1=1, d2=2, d3=3, d4=4, reduction=4),
        )

        self.heatmap_outputs_res1 = nn.Sequential(
            nn.Conv2d(features, num_landmarks, kernel_size=1, stride=1, padding=0)
        )

        if path:
            self.load(path)

    def forward(self, x):

        y1 = self.aspp(x)
        z1 = self.heatmap_outputs_res1(y1)

        return z1

class BackboneV2(torch.nn.Module):
    """
    Backbone network v2
    """

    def __init__(self, model, path=None, output_downsample=8, features=320):

        super(BackboneV2, self).__init__()

        self.pretrained, channels = _make_encoder(use_pretrained=True, model = model, output_downsample=output_downsample)

        self.output_channels = channels

        self.aspp = nn.Sequential(
            ASPP(in_ch=features, d1=1, d2=2, d3=3, d4=4, reduction=4),
        )

        if path:
            self.load(path)

    def forward(self, x):

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        dilated = self.aspp(layer_4)

        return dilated

class SceneHeadV2(torch.nn.Module):

    """
    Scene head network v2
    """

    def __init__(self, path=None, num_landmarks=200, features=320):

        super(SceneHeadV2, self).__init__()

        self.heatmap_outputs_res1 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(features*2, num_landmarks, kernel_size=1, stride=1, padding=0)
        )

        if path:
            self.load(path)

    def forward(self, x):

        z1 = self.heatmap_outputs_res1(x)

        return z1