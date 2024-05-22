import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BackboneV1(nn.Module):
    """
    Backbone network v1
    """

    def __init__(self, model, path=None, output_downsample=8):

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


class SceneHeadV1(nn.Module):

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
    
class BackboneV2(nn.Module):
    """
    Backbone network v2
    """

    def __init__(self, model, path=None, output_downsample=8):

        super(BackboneV2, self).__init__()

        self.pretrained, channels = _make_encoder(use_pretrained=True, model = model, output_downsample=output_downsample)

        self.output_channels = channels

        self.aspp = nn.Sequential(
            ASPP(in_ch=channels, d1=1, d2=2, d3=3, d4=4, reduction=4),
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
    
class SceneHeadV2(nn.Module):

    """
    Scene head network v2
    """

    def __init__(self, path=None, num_landmarks=200, features=320):

        super(SceneHeadV2, self).__init__()

        self.heatmap_outputs_res1 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(features*2, num_landmarks, kernel_size=1, stride=1, padding=0),
        )

        if path:
            self.load(path)

    def forward(self, x):

        z1 = self.heatmap_outputs_res1(x)

        return z1
    
class MultiHeadModel(nn.Module):

    """
    combined model
    """

    def __init__(self, bb_model, head_version, scenes, path=None, num_landmarks=200, features=320):
        super(MultiHeadModel, self).__init__()

        if "v2" in bb_model:
            self.bb = BackboneV2(bb_model)
        elif "v1" in bb_model:
            self.bb = BackboneV1(bb_model)

        self.single_scene = len(scenes) == 1
        self.scene = scenes[0]
        # bandaid fix
        scenes = ["scene1","scene2a","scene4a","scene5","scene6"]

        heads = {}
        if "v3" == head_version:
            for scene in scenes:
                heads[scene] = SceneHeadV3(features=features, num_landmarks=num_landmarks)
        elif "v2" == head_version:
            for scene in scenes:
                heads[scene] = SceneHeadV2(features=features, num_landmarks=num_landmarks)
        elif "v1" == head_version:
            for scene in scenes:
                heads[scene] = SceneHeadV1(features=features, num_landmarks=num_landmarks)

        self.heads = nn.ModuleDict(heads)

        if path:
            self.load(path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x, scene=None):

        bb_output = self.bb(x)

        if self.single_scene:
            head_output = self.heads[self.scene](bb_output)
        else:
            head_output = self.heads[scene](bb_output)

        return head_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SceneHeadV3(nn.Module):
    """
    Scene Head V3, more 1x1 convolutions
    """
    def __init__(self, path=None, num_landmarks=200, features=320):

        super(SceneHeadV3, self).__init__()

        downsample = nn.Sequential(
            nn.Conv2d(features, 512, 1),
            nn.BatchNorm2d(512)
        )

        self.heatmap_outputs_res1 = nn.Sequential(
            ResidualBlock(features, 512, downsample=downsample),
            nn.Conv2d(512,num_landmarks,1)
        )

        if path:
            self.load(path)

    def forward(self, x):

        z1 = self.heatmap_outputs_res1(x)

        return z1

class SceneHeadV4(nn.Module):
    """
    Scene Head V4, more 1x1 convolutions
    """
    def __init__(self, path=None, num_landmarks=200, features=320):

        super(SceneHeadV4, self).__init__()

        downsample = nn.Sequential(
            nn.Conv2d(features, 512, 1),
            nn.BatchNorm2d(512)
        )

        self.heatmap_outputs_res1 = nn.Sequential(
            ResidualBlock(features, 512, downsample=downsample),
            ResidualBlock(512, 512, downsample=None),
            nn.Conv2d(512,num_landmarks,1)
        )

        if path:
            self.load(path)

    def forward(self, x):

        z1 = self.heatmap_outputs_res1(x)

        return z1

class ACEBlock(nn.Module):
    def __init__(self, input_channels=2048, head_channels=512):
        super(ACEBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, head_channels , 1)
        self.conv2 = nn.Conv2d(head_channels, head_channels , 1)
        self.conv3 = nn.Conv2d(head_channels, head_channels , 1)

    def forward(self, x):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(res))
        res = F.relu(self.conv3(res))
        return res

class ACEHead(nn.Module):

    def __init__(self, input_channels=2048, num_landmarks=200):
        super(ACEHead, self).__init__()
        
        self.input_channels = input_channels

        self.num_landmarks = num_landmarks

        self.head_channels = 512

        self.block1 = ACEBlock(self.input_channels, self.head_channels)
        self.block2 = ACEBlock(self.head_channels, self.head_channels)

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels , 1)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels , 1)
        self.fc3 = nn.Conv2d(self.head_channels, self.num_landmarks , 1)

        self.downsample = nn.Conv2d(self.input_channels, self.head_channels, 1)

    def forward(self, x):
        res = self.block1(x)

        x = self.downsample(x) + res

        res = self.block2(x)

        x = x + res

        res = F.relu(self.fc1(x))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)

        return res

class FrozenBackBoneModel(nn.Module):

    def __init__(self, bb=None, head_model="ace", path=None):
        super(FrozenBackBoneModel, self).__init__()
        
        if bb == None:
            self.bb = BackboneV2
        else:
            self.bb = bb

        for param in bb.parameters():
            param.requires_grad = False

        if head_model == "ace":
            self.head = ACEHead()
        elif head_model == "simple":
            self.head = SceneHeadV1(features=2048,num_landmarks=200)

        if path:
            self.load(path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        with torch.no_grad():
            features = self.bb(x)
        output = self.head(features)
        return output