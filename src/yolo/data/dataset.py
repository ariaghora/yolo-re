"""YOLO dataset for object detection training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yolo.data.config import DataConfig

if TYPE_CHECKING:
    from yolo.data.transforms import Compose


class YOLODataset(Dataset[tuple[Tensor, Tensor, str, tuple[int, int]]]):
    """Dataset for YOLO object detection.

    Expects COCO-style directory structure:
        images/train/xxx.jpg
        labels/train/xxx.txt

    Label format (one line per object):
        class_id x_center y_center width height
    All coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        path: Path | str,
        img_size: int = 640,
        transforms: Compose | None = None,
        cache_images: bool = False,
    ):
        """Initialize dataset.

        Args:
            path: Path to images directory or .txt file with image paths
            img_size: Target image size
            transforms: Transform pipeline to apply
            cache_images: Cache images in RAM for faster training
        """
        self.path = Path(path)
        self.img_size = img_size
        self.transforms = transforms

        self.im_files = self._get_image_files()
        self.label_files = self._img2label_paths(self.im_files)
        self.labels = self._load_labels()

        self.n = len(self.im_files)
        self.indices = list(range(self.n))

        self.imgs: list[np.ndarray | None] = [None] * self.n
        if cache_images:
            self._cache_images()

    def _get_image_files(self) -> list[Path]:
        """Get list of image files."""
        if self.path.is_file() and self.path.suffix == ".txt":
            with open(self.path) as f:
                return [Path(line.strip()) for line in f if line.strip()]

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = []
        for ext in extensions:
            files.extend(self.path.rglob(f"*{ext}"))
            files.extend(self.path.rglob(f"*{ext.upper()}"))
        return sorted(files)

    def _img2label_paths(self, img_paths: list[Path]) -> list[Path]:
        """Convert image paths to label paths."""
        label_paths = []
        for p in img_paths:
            parts = list(p.parts)
            for i, part in enumerate(parts):
                if part == "images":
                    parts[i] = "labels"
                    break
            label_path = Path(*parts).with_suffix(".txt")
            label_paths.append(label_path)
        return label_paths

    def _load_labels(self) -> list[np.ndarray]:
        """Load all labels from disk."""
        labels = []
        for label_file in self.label_files:
            if label_file.exists():
                with open(label_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if x]
                    lb = np.array(lb, dtype=np.float64)
            else:
                lb = np.zeros((0, 5), dtype=np.float64)
            labels.append(lb)
        return labels

    def _cache_images(self) -> None:
        """Cache images in RAM."""
        for i in range(self.n):
            self.imgs[i] = cv2.imread(str(self.im_files[i]))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, tuple[int, int]]:
        """Get a training sample.

        Returns:
            img: Image tensor, shape (3, H, W) in RGB format, normalized to [0, 1]
            labels: Labels tensor, shape (N, 6) as [0, class_id, x, y, w, h]
            path: Image file path
            shapes: Original image shape (h, w)
        """
        from yolo.data.transforms import Sample

        img, (h0, w0), _ = self._load_image(index)
        labels = self.labels[index].copy()

        sample = Sample(
            img=img,
            labels=labels,
            img_size=self.img_size,
            original_shape=(h0, w0),
        )

        if self.transforms:
            sample = self.transforms(sample)

        img_tensor, labels_out = self._to_tensor(sample.img, sample.labels)
        return img_tensor, labels_out, str(self.im_files[index]), sample.original_shape

    def _load_image(self, i: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load image by index.

        Returns:
            img: Loaded image
            (h0, w0): Original dimensions
            (h, w): Current dimensions
        """
        img = self.imgs[i]
        if img is None:
            path = self.im_files[i]
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {path}")
        h0, w0 = img.shape[:2]
        return img, (h0, w0), (h0, w0)

    def _to_tensor(self, img: np.ndarray, labels: np.ndarray) -> tuple[Tensor, Tensor]:
        """Convert image and labels to tensors."""
        nl = len(labels)
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float() / 255.0

        return img_tensor, labels_out


def collate_fn(
    batch: list[tuple[Tensor, Tensor, str, tuple[int, int]]],
) -> tuple[Tensor, Tensor, tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Collate function for DataLoader.

    Handles variable number of targets per image by concatenating
    all targets and adding batch index.
    """
    imgs, labels, paths, shapes = zip(*batch)
    for i, lb in enumerate(labels):
        lb[:, 0] = i
    return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes


def create_dataloader(
    config: DataConfig,
    train: bool = True,
) -> DataLoader[tuple[Tensor, Tensor, str, tuple[int, int]]]:
    """Create a DataLoader from config.

    Args:
        config: Data configuration
        train: Whether this is for training (enables augmentation)

    Returns:
        DataLoader instance
    """
    from yolo.data.transforms import default_train_transforms, default_val_transforms

    path = config.train_path if train else config.val_path
    if path is None:
        raise ValueError("Path not specified in config")

    dataset = YOLODataset(
        path=path,
        img_size=config.img_size,
        transforms=None,  # Set after creation for dataset access
        cache_images=config.cache_images,
    )

    if train:
        aug = config.augment
        transforms = default_train_transforms(
            dataset,
            mosaic=aug.mosaic,
            mixup=aug.mixup,
            degrees=aug.degrees,
            translate=aug.translate,
            scale=aug.scale,
            shear=aug.shear,
            perspective=aug.perspective,
            hsv_h=aug.hsv_h,
            hsv_s=aug.hsv_s,
            hsv_v=aug.hsv_v,
            flipud=aug.flipud,
            fliplr=aug.fliplr,
        )
    else:
        transforms = default_val_transforms()

    dataset.transforms = transforms

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=train,
    )
