"""GELAN (Generalized ELAN) blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor

from yolo.blocks.conv import Conv
from yolo.blocks.csp import RepNCSP


@dataclass
class RepNCSPELAN4Config:
    """Configuration for RepNCSPELAN4 block."""

    in_channels: int
    out_channels: int
    hidden_channels: int
    block_channels: int
    num_repeats: int = 1


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN block with RepNCSP.

    The main building block of YOLOv9/GELAN architecture.
    Splits input, processes through RepNCSP blocks, and concatenates.

    Reference: _reference/yolov9/models/common.py::RepNCSPELAN4
    """

    Config = RepNCSPELAN4Config

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        block_channels: int,
        num_repeats: int = 1,
    ):
        super().__init__()
        self.conv_in = Conv(in_channels, hidden_channels, 1, 1)
        self.block1 = nn.Sequential(
            RepNCSP(hidden_channels // 2, block_channels, num_repeats),
            Conv(block_channels, block_channels, 3, 1),
        )
        self.block2 = nn.Sequential(
            RepNCSP(block_channels, block_channels, num_repeats),
            Conv(block_channels, block_channels, 3, 1),
        )
        self.conv_out = Conv(hidden_channels + (2 * block_channels), out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.conv_in(x).chunk(2, 1))
        y.append(self.block1(y[-1]))
        y.append(self.block2(y[-1]))
        return self.conv_out(torch.cat(y, 1))

    @classmethod
    def from_config(cls, cfg: RepNCSPELAN4Config) -> "RepNCSPELAN4":
        return cls(**asdict(cfg))
