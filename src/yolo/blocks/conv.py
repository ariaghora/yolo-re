"""Convolution blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch.nn as nn
from torch import Tensor


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    """Calculate 'same' padding for a convolution.

    Reference: _reference/yolov9/models/common.py::autopad
    """
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    match name:
        case "silu":
            return nn.SiLU()
        case "relu":
            return nn.ReLU()
        case "leaky_relu":
            return nn.LeakyReLU(0.1)
        case "hardswish":
            return nn.Hardswish()
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unknown activation: {name}")


@dataclass
class ConvConfig:
    """Configuration for Conv block."""

    in_channels: int
    out_channels: int
    kernel_size: int = 1
    stride: int = 1
    padding: int | None = None
    groups: int = 1
    dilation: int = 1
    activation: str = "silu"


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation.

    Reference: _reference/yolov9/models/common.py::Conv
    """

    Config = ConvConfig

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    @classmethod
    def from_config(cls, cfg: ConvConfig) -> "Conv":
        return cls(**asdict(cfg))


@dataclass
class RepConvConfig:
    """Configuration for RepConv block."""

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    groups: int = 1
    activation: str = "silu"


class RepConv(nn.Module):
    """Re-parameterizable convolution block.

    During training, uses parallel 3x3 and 1x1 convolutions.

    Reference: _reference/yolov9/models/common.py::RepConvN
    """

    Config = RepConvConfig

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        assert kernel_size == 3 and padding == 1, "RepConv only supports 3x3 kernels"

        self.conv1 = Conv(
            in_channels, out_channels, kernel_size, stride, padding, groups, activation="none"
        )
        self.conv2 = Conv(
            in_channels, out_channels, 1, stride, padding=0, groups=groups, activation="none"
        )
        self.act = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))

    @classmethod
    def from_config(cls, cfg: RepConvConfig) -> "RepConv":
        return cls(**asdict(cfg))
