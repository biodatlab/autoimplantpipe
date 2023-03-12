"""
This code is adapted from the original 3DUnetCNN.
Reference: https://github.com/ellisdg/3DUnetCNN
"""
import torch
import torch.nn as nn
from torch.nn.common_types import _size_3_t
from typing import List


def conv1x1x1(chin: int, chout: int):
    return nn.Conv3d(chin, chout, 1, bias=False)


def conv3x3x3(chin: int, chout: int, stride: _size_3_t = 1):
    return nn.Conv3d(chin, chout, 3, stride=stride, padding=1, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int):
        super(ConvBlock, self).__init__()
        num_groups = in_planes if in_planes < 8 else 8
        self.norm1 = nn.GroupNorm(num_groups, in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3x3(in_planes, planes)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_planes, planes)
        self.conv2 = ConvBlock(planes, planes)
        self.sample = conv1x1x1(in_planes, planes) if in_planes != planes else None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.sample:
            identity = self.sample(identity)
        x += identity
        return x


class UNetLayer(nn.Module):
    def __init__(
        self, n_blocks: int, in_planes: int, planes: int, dropout: float = None
    ):
        super(UNetLayer, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ResidualBlock(in_planes, planes))
            in_planes = planes
        self.dropout = nn.Dropout3d(dropout, inplace=True) if dropout else None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        base_width: int = 32,
        layer_blocks: List[int] = [2, 2, 2, 2, 2],
        feature_dilation: int = 2,
        stride: int = 2,
        dropout: float = 0.2,
    ):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        in_width = n_features

        for i, n_blocks in enumerate(layer_blocks):
            out_width = base_width * (feature_dilation**i)
            self.layers.append(UNetLayer(n_blocks, in_width, out_width, dropout))
            if i != len(layer_blocks) - 1:
                self.layers.append(conv3x3x3(out_width, out_width, stride))
            in_width = out_width
            dropout = None

    def forward(self, x):
        outputs = list()
        for layer in self.layers:
            x = layer(x)
            if type(layer) == UNetLayer:
                outputs.insert(0, x)  # residual
        return outputs


class UNetDecoder(nn.Module):
    def __init__(
        self,
        base_width: int = 32,
        layer_blocks: List[int] = [1, 1, 1, 1, 1],
        upsampling_scale: int = 2,
        feature_dilation: int = 2,
    ):
        super(UNetDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.feature_reduction_scale = feature_dilation
        self.upsampling_scale = upsampling_scale
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            if depth != 0:
                out_width = int(
                    base_width * (self.feature_reduction_scale ** (depth - 1))
                )
                in_width = out_width * self.feature_reduction_scale
                if depth != len(layer_blocks) - 1:
                    in_width *= 2
                self.layers.append(UNetLayer(n_blocks, in_width, in_width))
                self.layers.append(conv1x1x1(in_width, out_width))
            else:
                self.layers.append(UNetLayer(n_blocks, base_width * 2, base_width))

    def forward(self, inputs):
        x = inputs.pop(0)
        for layer in self.layers:
            x = layer(x)
            if type(layer) == nn.Conv3d:
                x = nn.functional.interpolate(
                    x, scale_factor=self.upsampling_scale, mode="trilinear"
                )
                x = torch.cat((x, inputs.pop(0)), 1)  # residual
        return x


class UNetAutoEncoder(nn.Module):
    def __init__(self, base_width: int = 32, n_features: int = 1):
        super(UNetAutoEncoder, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.final_convolution = conv1x1x1(base_width, n_features)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        x = self.activation(x)
        return x


class AutoImplantUNet(UNetAutoEncoder):
    def __init__(self):
        super(AutoImplantUNet, self).__init__()

    def load_pretrained_model(self, model_filename: str):
        """
        Load pretrain model from h5 or pth file.
        """
        pretrained_dict = torch.load(model_filename)

        if "module.final_convolution.weight" not in pretrained_dict:
            return self.load_state_dict(pretrained_dict)

        state_dict = self.state_dict()

        # convert baseline pretrained data to current structure
        state_dict["final_convolution.weight"] = pretrained_dict[
            "module.final_convolution.weight"
        ]
        for i in range(5):
            for key in pretrained_dict.keys():
                for t in ["encoder", "decoder"]:
                    if key.startswith(f"module.{t}.layers.{i}"):
                        new_key = key.replace(
                            f"module.{t}.layers.{i}", f"{t}.layers.{i*2}"
                        )
                        state_dict[new_key] = pretrained_dict[key]
        for i in range(4):
            state_dict[f"encoder.layers.{(i*2)+1}.weight"] = pretrained_dict[
                f"module.encoder.downsampling_convolutions.{i}.weight"
            ]
            state_dict[f"decoder.layers.{(i*2)+1}.weight"] = pretrained_dict[
                f"module.decoder.pre_upsampling_blocks.{i}.weight"
            ]

        self.load_state_dict(state_dict)
