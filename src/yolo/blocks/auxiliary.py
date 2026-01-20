"""Auxiliary branch blocks for YOLOv9.

These blocks are used in the multi-level reversible auxiliary branch.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yolo.blocks.conv import autopad


@dataclass
class CBLinearConfig:
    """Configuration for CBLinear block."""

    in_channels: int
    out_channels_list: list[int]
    kernel_size: int = 1
    stride: int = 1
    padding: int | None = None
    groups: int = 1


class CBLinear(nn.Module):
    """Channel-broadcast linear block.

    Projects input to multiple output channel sizes, splitting the output.

    Reference: _reference/yolov9/models/common.py::CBLinear
    """

    Config = CBLinearConfig

    def __init__(
        self,
        in_channels: int,
        out_channels_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ):
        super().__init__()
        self.out_channels_list = out_channels_list
        self.conv = nn.Conv2d(
            in_channels,
            sum(out_channels_list),
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=True,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.conv(x).split(self.out_channels_list, dim=1)

    @classmethod
    def from_config(cls, cfg: CBLinearConfig) -> "CBLinear":
        return cls(**asdict(cfg))


@dataclass
class CBFuseConfig:
    """Configuration for CBFuse block."""

    idx: list[int]


class CBFuse(nn.Module):
    """Channel-broadcast fuse block.

    Fuses multiple feature maps by interpolating to target size and summing.

    Reference: _reference/yolov9/models/common.py::CBFuse
    """

    Config = CBFuseConfig

    def __init__(self, idx: list[int]):
        super().__init__()
        self.idx = idx

    def forward(self, inputs: list[tuple[Tensor, ...] | Tensor]) -> Tensor:
        """Fuse CBLinear outputs with a target tensor.

        Args:
            inputs: List where first N-1 elements are tuples from CBLinear blocks,
                and last element is the target tensor.

        Returns:
            Sum of interpolated features and target.
        """
        cb_outputs = inputs[:-1]
        target = inputs[-1]
        if not isinstance(target, Tensor):
            raise TypeError("Last input must be target Tensor, not CBLinear tuple")

        target_size = target.shape[2:]
        res = [
            F.interpolate(cb_out[self.idx[i]], size=target_size, mode="nearest")
            for i, cb_out in enumerate(cb_outputs)
        ]
        return torch.sum(torch.stack([*res, target]), dim=0)

    @classmethod
    def from_config(cls, cfg: CBFuseConfig) -> "CBFuse":
        return cls(**asdict(cfg))
