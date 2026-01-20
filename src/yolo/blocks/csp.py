"""CSP (Cross Stage Partial) blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor

from yolo.blocks.bottleneck import RepNBottleneck
from yolo.blocks.conv import Conv


@dataclass
class RepNCSPConfig:
    """Configuration for RepNCSP block."""

    in_channels: int
    out_channels: int
    num_repeats: int = 1
    shortcut: bool = True
    groups: int = 1
    expansion_ratio: float = 0.5


class RepNCSP(nn.Module):
    """CSP Bottleneck with 3 convolutions using RepNBottleneck.

    Reference: _reference/yolov9/models/common.py::RepNCSP
    """

    Config = RepNCSPConfig

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_repeats: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion_ratio)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1)
        self.bottlenecks = nn.Sequential(
            *[
                RepNBottleneck(
                    hidden_channels, hidden_channels, shortcut, groups, expansion_ratio=1.0
                )
                for _ in range(num_repeats)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(torch.cat((self.bottlenecks(self.conv1(x)), self.conv2(x)), dim=1))

    @classmethod
    def from_config(cls, cfg: RepNCSPConfig) -> "RepNCSP":
        return cls(**asdict(cfg))
