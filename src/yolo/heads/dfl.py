"""Distribution Focal Loss decoder.

DFL predicts box coordinates as discrete probability distributions over bins,
then decodes to continuous values via expected value.

Reference: _reference/yolov9/models/common.py
"""

import torch
import torch.nn as nn
from torch import Tensor


class DFL(nn.Module):
    """Decodes DFL-style box predictions to continuous coordinates.

    Computes expected value: sum(softmax(logits) * [0, 1, 2, ..., num_bins-1]).

    Reference: _reference/yolov9/models/common.py::DFL
    """

    def __init__(self, num_bins: int = 16):
        """Initialize DFL decoder.

        Args:
            num_bins: Number of discrete bins (reg_max). Default 16.
        """
        super().__init__()
        self.num_bins = num_bins
        # Conv with fixed weights [0, 1, 2, ..., num_bins-1]
        self.conv = nn.Conv2d(num_bins, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(num_bins, dtype=torch.float).view(
            1, num_bins, 1, 1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, channels, anchors] where
               channels = 4 * num_bins (4 box coordinates).

        Returns:
            Box coordinates of shape [batch, 4, anchors].
        """
        batch, channels, anchors = x.shape
        # Reshape to [batch, 4, num_bins, anchors], softmax over bins, compute expected value
        x = x.view(batch, 4, self.num_bins, anchors).transpose(2, 1).softmax(1)
        # Conv computes weighted sum: [0,1,2,...] dot softmax -> expected bin value
        return self.conv(x).view(batch, 4, anchors)
