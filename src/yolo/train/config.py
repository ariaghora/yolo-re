"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Model and data config are passed separately to Trainer.
    """

    # Training
    epochs: int = 100

    # Optimizer (used if optimizer not passed to Trainer directly)
    lr: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Scheduler
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    lrf: float = 0.01  # final lr = lr * lrf

    # Checkpointing
    output_dir: Path | str = "runs/train"
    save_period: int = -1  # -1 means only best/last

    # Validation
    val_period: int = 1  # Validate every N epochs (1 = every epoch)

    # Device
    device: str = "auto"  # auto, cuda, mps, cpu
    amp: bool = True

    # Logging
    log_interval: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
