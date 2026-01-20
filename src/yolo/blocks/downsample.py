"""Downsampling blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yolo.blocks.conv import Conv


@dataclass
class ADownConfig:
    """Configuration for ADown block."""

    in_channels: int
    out_channels: int


class ADown(nn.Module):
    """Average pooling downsample block.

    Combines average pooling with max pooling paths for downsampling.

    Reference: _reference/yolov9/models/common.py::ADown
    """

    Config = ADownConfig

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_out = out_channels // 2
        self.conv_stride = Conv(in_channels // 2, half_out, 3, 2, 1)
        self.conv_pool = Conv(in_channels // 2, half_out, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(x, 2, 1, 0, ceil_mode=True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.conv_stride(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.conv_pool(x2)
        return torch.cat((x1, x2), 1)

    @classmethod
    def from_config(cls, cfg: ADownConfig) -> "ADown":
        return cls(**asdict(cfg))
