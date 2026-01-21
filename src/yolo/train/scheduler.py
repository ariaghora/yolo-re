"""Learning rate scheduler with warmup for YOLO training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay.

    This implements the standard YOLO training schedule:
    1. Linear warmup from warmup_bias_lr (for biases) and 0 (for others) to lr0
    2. Cosine decay from lr0 to lr0 * lrf

    The scheduler also handles momentum warmup for SGD optimizers.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        steps_per_epoch: int,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        lrf: float = 0.01,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to schedule.
            total_epochs: Total number of training epochs.
            steps_per_epoch: Number of batches per epoch.
            warmup_epochs: Number of warmup epochs (can be fractional).
            warmup_momentum: Starting momentum during warmup.
            warmup_bias_lr: Starting lr for biases during warmup.
            lrf: Final learning rate factor (final_lr = lr0 * lrf).
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.lrf = lrf

        # Store initial learning rates and momentum
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.base_momentum = self._get_momentum()

        self.current_step = 0

    def _get_momentum(self) -> float:
        """Get current momentum from optimizer."""
        group = self.optimizer.param_groups[0]
        if "momentum" in group:
            return group["momentum"]
        if "betas" in group:
            return group["betas"][0]
        return 0.9

    def _set_momentum(self, momentum: float) -> None:
        """Set momentum for all parameter groups."""
        for group in self.optimizer.param_groups:
            if "momentum" in group:
                group["momentum"] = momentum
            elif "betas" in group:
                group["betas"] = (momentum, group["betas"][1])

    def step(self) -> None:
        """Update learning rate and momentum for current step."""
        self.current_step += 1
        step = self.current_step

        if step <= self.warmup_steps:
            # Linear warmup phase
            xi = step / self.warmup_steps

            for i, group in enumerate(self.optimizer.param_groups):
                # Bias group (index 2) starts from warmup_bias_lr
                # Other groups start from 0
                if i == 2:  # bias group
                    group["lr"] = self.warmup_bias_lr + (
                        self.base_lrs[i] - self.warmup_bias_lr
                    ) * xi
                else:
                    group["lr"] = self.base_lrs[i] * xi

            # Warmup momentum
            self._set_momentum(
                self.warmup_momentum
                + (self.base_momentum - self.warmup_momentum) * xi
            )
        else:
            # Cosine decay phase
            total_steps = self.total_epochs * self.steps_per_epoch
            progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
            progress = min(progress, 1.0)

            # Cosine annealing: lr = lrf + (1 - lrf) * 0.5 * (1 + cos(pi * progress))
            decay = self.lrf + (1 - self.lrf) * 0.5 * (1 + math.cos(math.pi * progress))

            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * decay

    def get_lr(self) -> list[float]:
        """Get current learning rates for all parameter groups."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        """Alias for get_lr for compatibility."""
        return self.get_lr()

    @property
    def epoch(self) -> float:
        """Get current epoch (fractional)."""
        return self.current_step / self.steps_per_epoch


def one_cycle_lr(epoch: int, total_epochs: int, lrf: float = 0.01) -> float:
    """Compute one-cycle learning rate multiplier.

    This is a simpler epoch-based version for use with torch schedulers.

    Args:
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of epochs.
        lrf: Final learning rate factor.

    Returns:
        Learning rate multiplier for this epoch.
    """
    # Cosine annealing
    return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
