"""SPPELAN block.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor

from yolo.blocks.conv import Conv


@dataclass
class SPPELANConfig:
    """Configuration for SPPELAN block."""

    in_channels: int
    out_channels: int
    hidden_channels: int


class SPPELAN(nn.Module):
    """SPP with ELAN structure.

    Applies spatial pyramid pooling using max pools of kernel size 5,
    then concatenates and projects.

    Reference: _reference/yolov9/models/common.py::SPPELAN
    """

    Config = SPPELANConfig

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.conv_in = Conv(in_channels, hidden_channels, 1, 1)
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(5, 1, 2)
        self.pool3 = nn.MaxPool2d(5, 1, 2)
        self.conv_out = Conv(4 * hidden_channels, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = [self.conv_in(x)]
        y.append(self.pool1(y[-1]))
        y.append(self.pool2(y[-1]))
        y.append(self.pool3(y[-1]))
        return self.conv_out(torch.cat(y, 1))

    @classmethod
    def from_config(cls, cfg: SPPELANConfig) -> "SPPELAN":
        return cls(**asdict(cfg))
