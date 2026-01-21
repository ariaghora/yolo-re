"""Data configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml


class CacheMode(Enum):
    """Image caching strategy."""

    NONE = "none"  # No caching, load from disk each time
    RAM = "ram"  # Cache resized images in RAM
    DISK = "disk"  # Cache resized images as .npy files


AugmentPreset = Literal["full", "light", "minimal"]

# Preset definitions for augmentation
_AUGMENT_PRESETS: dict[AugmentPreset, dict[str, float | tuple[float, float]]] = {
    # Full augmentation for training from scratch (matches hyp.scratch-high.yaml)
    "full": {
        "mosaic": 1.0,
        "mosaic_scale": (0.5, 1.5),
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.9,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
    },
    # Light augmentation for fine-tuning pretrained weights
    "light": {
        "mosaic": 0.5,
        "mosaic_scale": (0.8, 1.2),
        "mixup": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
    },
    # Minimal augmentation for debugging or very short fine-tuning
    "minimal": {
        "mosaic": 0.0,
        "mosaic_scale": (1.0, 1.0),
        "mixup": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
    },
}


@dataclass
class AugmentConfig:
    """Augmentation configuration.

    Use preset="full" for training from scratch (matches hyp.scratch-high.yaml).
    Use preset="light" for fine-tuning pretrained weights.
    Use preset="minimal" for debugging or when augmentation hurts performance.

    Individual parameters can override preset defaults.
    """

    preset: AugmentPreset = "full"

    # Mosaic (None = use preset default)
    mosaic: float | None = None
    mosaic_scale: tuple[float, float] | None = None
    mixup: float | None = None

    # HSV
    hsv_h: float | None = None
    hsv_s: float | None = None
    hsv_v: float | None = None

    # Geometric
    degrees: float | None = None
    translate: float | None = None
    scale: float | None = None
    shear: float | None = None
    perspective: float | None = None

    # Flips
    flipud: float | None = None
    fliplr: float | None = None

    def __post_init__(self) -> None:
        """Apply preset defaults for any None values."""
        if self.preset not in _AUGMENT_PRESETS:
            valid = list(_AUGMENT_PRESETS)
            raise ValueError(f"Unknown preset: {self.preset}. Choose from: {valid}")

        defaults = _AUGMENT_PRESETS[self.preset]
        for key, val in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, val)

    @classmethod
    def from_preset(cls, preset: AugmentPreset) -> AugmentConfig:
        """Create config from a preset name."""
        return cls(preset=preset)


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_path: Path | str
    val_path: Path | str | None = None
    num_classes: int = 80
    class_names: list[str] = field(default_factory=list)

    # Image settings
    img_size: int = 640
    batch_size: int = 16
    workers: int = 8

    # Augmentation
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    # Caching and rect training
    cache: CacheMode = CacheMode.NONE
    rect: bool = False
    stride: int = 32  # For rect training shape alignment

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        if self.val_path is not None:
            self.val_path = Path(self.val_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
