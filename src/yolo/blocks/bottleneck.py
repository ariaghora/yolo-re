"""Bottleneck blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch.nn as nn
from torch import Tensor

from yolo.blocks.conv import Conv, RepConv


@dataclass
class RepNBottleneckConfig:
    """Configuration for RepNBottleneck block."""

    in_channels: int
    out_channels: int
    shortcut: bool = True
    groups: int = 1
    kernel_sizes: tuple[int, int] = (3, 3)
    expansion_ratio: float = 0.5


class RepNBottleneck(nn.Module):
    """Bottleneck with RepConv in the first convolution.

    Reference: _reference/yolov9/models/common.py::RepNBottleneck
    """

    Config = RepNBottleneckConfig

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_sizes: tuple[int, int] = (3, 3),
        expansion_ratio: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion_ratio)
        self.conv1 = RepConv(in_channels, hidden_channels, kernel_sizes[0], 1)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_sizes[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out

    @classmethod
    def from_config(cls, cfg: RepNBottleneckConfig) -> "RepNBottleneck":
        return cls(**asdict(cfg))
