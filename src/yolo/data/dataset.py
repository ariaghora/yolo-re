"""YOLO dataset for object detection training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from yolo.data.config import CacheMode, DataConfig

if TYPE_CHECKING:
    from yolo.data.transforms import Compose


def seed_worker(worker_id: int) -> None:
    """Set worker seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        cache: CacheMode = CacheMode.NONE,
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.0,
    ):
        """Initialize dataset.

        Args:
            path: Path to images directory or .txt file with image paths
            img_size: Target image size
            transforms: Transform pipeline to apply
            cache: Caching strategy (NONE, RAM, or DISK)
            rect: Use rectangular training (variable batch shapes by aspect ratio)
            batch_size: Batch size (needed for rect training)
            stride: Stride for shape alignment in rect training
            pad: Padding factor for rect training
        """
        self.path = Path(path)
        self.img_size = img_size
        self.transforms = transforms
        self.cache = cache
        self.rect = rect
        self.stride = stride

        self.im_files = self._get_image_files()
        self.label_files = self._img2label_paths(self.im_files)
        self.npy_files = [f.with_suffix(".npy") for f in self.im_files]
        self.labels = self._load_labels()

        self.n = len(self.im_files)
        self.indices = list(range(self.n))

        # Load image shapes for rect training
        self.shapes = self._load_shapes()

        # Setup rect training (sort by aspect ratio)
        self.batch: np.ndarray | None = None
        self.batch_shapes: np.ndarray | None = None
        if rect:
            self._setup_rect(batch_size, pad)

        # Initialize image cache
        self.imgs: list[np.ndarray | None] = [None] * self.n
        if cache == CacheMode.DISK:
            self._cache_images_to_disk()
        elif cache == CacheMode.RAM:
            self._cache_images_to_ram()

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

    def _load_shapes(self) -> np.ndarray:
        """Load image shapes (h, w) for all images."""
        shapes = []
        for f in self.im_files:
            img = cv2.imread(str(f))
            if img is not None:
                shapes.append(img.shape[:2])
            else:
                shapes.append((self.img_size, self.img_size))
        return np.array(shapes)

    def _setup_rect(self, batch_size: int, pad: float) -> None:
        """Setup rectangular training by sorting images by aspect ratio."""
        # Sort by aspect ratio
        ar = self.shapes[:, 0] / self.shapes[:, 1]  # h/w
        irect = ar.argsort()

        # Reorder everything by aspect ratio
        self.im_files = [self.im_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.npy_files = [self.npy_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = self.shapes[irect]
        ar = ar[irect]

        # Compute batch indices and shapes
        bi = np.floor(np.arange(self.n) / batch_size).astype(int)
        nb = bi[-1] + 1 if self.n > 0 else 0
        self.batch = bi

        # Compute batch shapes
        self.batch_shapes = np.zeros((nb, 2), dtype=np.float64)
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                self.batch_shapes[i] = [maxi, 1]
            elif mini > 1:
                self.batch_shapes[i] = [1, 1 / mini]
            else:
                self.batch_shapes[i] = [1, 1]

        self.batch_shapes = (
            np.ceil(self.batch_shapes * self.img_size / self.stride + pad).astype(int) * self.stride
        )

    def _cache_images_to_ram(self) -> None:
        """Cache images in RAM."""
        desc = f"Caching images to RAM ({self.path.name})"
        for i in tqdm(range(self.n), desc=desc):
            self.imgs[i] = cv2.imread(str(self.im_files[i]))

    def _cache_images_to_disk(self) -> None:
        """Cache resized images as .npy files."""
        desc = f"Caching images to disk ({self.path.name})"
        for i in tqdm(range(self.n), desc=desc):
            npy = self.npy_files[i]
            if not npy.exists():
                img = cv2.imread(str(self.im_files[i]))
                if img is not None:
                    h0, w0 = img.shape[:2]
                    r = self.img_size / max(h0, w0)
                    if r != 1:
                        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
                    np.save(npy, img)

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

        # Use batch shape for rect training
        img_size = self.img_size
        if self.rect and self.batch is not None and self.batch_shapes is not None:
            img_size = tuple(self.batch_shapes[self.batch[index]])  # type: ignore[assignment]

        sample = Sample(
            img=img,
            labels=labels,
            img_size=img_size if isinstance(img_size, int) else img_size[0],
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
        # Check RAM cache first
        img = self.imgs[i]

        if img is None:
            # Check disk cache
            npy = self.npy_files[i]
            if npy.exists():
                img = np.load(npy)
                h0, w0 = self.shapes[i]
                return img, (h0, w0), img.shape[:2]

            # Load from image file
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

    # Rect training only for validation (must maintain order)
    rect = config.rect and not train

    dataset = YOLODataset(
        path=path,
        img_size=config.img_size,
        transforms=None,  # Set after creation for dataset access
        cache=config.cache,
        rect=rect,
        batch_size=config.batch_size,
        stride=config.stride,
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

    # Worker seeding for reproducibility
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train and not rect,  # No shuffle for rect training
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )
