"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p))

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(
        self, in_channels1: int, in_channels2: int, out_channels: int, dropout_p: float, bilinear: bool = True
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["n_class"]
        self.bilinear = self.params["bilinear"]
        self.dropout = self.params["dropout"]
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["n_class"]
        self.bilinear = self.params["bilinear"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature: List[Tensor]) -> Tensor:
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


def Dropout(x: Tensor, p: float = 0.3) -> Tensor:
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x: Tensor) -> Tensor:
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range: float = 0.3) -> None:
        super().__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x: Tensor) -> Tensor:
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns: int, n_class: int) -> None:
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [64, 128, 256, 512, 1024],
            "dropout": [0, 0, 0, 0, 0],  # [0.05, 0.1, 0.2, 0.3, 0.5],
            "n_class": n_class,
            "bilinear": False,
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x: Tensor) -> Tensor:
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class small_UNet(nn.Module):
    def __init__(self, in_chns: int, n_class: int) -> None:
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [8, 16, 32, 64, 128],
            "dropout": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_class": n_class,
            "bilinear": False,
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x: Tensor) -> Tensor:
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet_CCT(nn.Module):
    def __init__(self, in_chns: int, n_class: int) -> None:
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [64, 128, 256, 512, 1024],
            "dropout": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_class": n_class,
            "bilinear": False,
        }
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class small_UNet_CCT(nn.Module):
    def __init__(self, in_chns: int, n_class: int) -> None:
        super().__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [8, 16, 32, 64, 128],
            "dropout": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_class": n_class,
            "bilinear": False,
        }
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


if __name__ == "main":
    input = torch.randn(1, 1, 32, 32)
    print(UNet(1, 3)(input).shape)
