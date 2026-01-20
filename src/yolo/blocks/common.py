"""Common utility blocks.

Reference: _reference/yolov9/models/common.py
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ConcatConfig:
    """Configuration for Concat block."""

    dimension: int = 1


class Concat(nn.Module):
    """Concatenate a list of tensors along a dimension.

    Reference: _reference/yolov9/models/common.py::Concat
    """

    Config = ConcatConfig

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.cat(x, self.dimension)

    @classmethod
    def from_config(cls, cfg: ConcatConfig) -> "Concat":
        return cls(**asdict(cfg))


class Silence(nn.Module):
    """Identity/placeholder module that returns input unchanged.

    Reference: _reference/yolov9/models/common.py::Silence
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
