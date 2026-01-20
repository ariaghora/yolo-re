"""Data configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CacheMode(Enum):
    """Image caching strategy."""

    NONE = "none"  # No caching, load from disk each time
    RAM = "ram"  # Cache resized images in RAM
    DISK = "disk"  # Cache resized images as .npy files


@dataclass
class AugmentConfig:
    """Augmentation configuration.

    Defaults match reference YOLOv9 hyp.scratch-high.yaml.
    """

    # Mosaic
    mosaic: float = 1.0
    mosaic_scale: tuple[float, float] = (0.5, 1.5)
    mixup: float = 0.15  # Reference default

    # HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # Geometric
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.9  # Reference default
    shear: float = 0.0
    perspective: float = 0.0

    # Flips
    flipud: float = 0.0
    fliplr: float = 0.5


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
