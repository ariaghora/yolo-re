"""Exponential Moving Average for model weights.

Reference: _reference/yolov9/utils/torch_utils.py::ModelEMA
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model weights.

    Keeps a moving average of model parameters and buffers. The EMA weights
    are used for validation and final checkpoint, providing smoother and
    typically better-performing weights than the raw trained weights.

    The decay ramps up from 0 to `decay` over `tau` steps, which helps
    stabilize early training.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000,
        updates: int = 0,
    ):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = decay
        self.tau = tau
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _get_decay(self) -> float:
        """Get current decay value (ramps up over tau steps)."""
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from model weights."""
        self.updates += 1
        d = self._get_decay()

        model_sd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * model_sd[k].detach()

    def state_dict(self) -> dict[str, Any]:
        """Return EMA state for checkpointing."""
        return {
            "ema_state_dict": self.ema.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.ema.load_state_dict(state["ema_state_dict"])
        self.updates = state["updates"]
