"""Composable transforms for YOLO training."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import albumentations as A
import numpy as np

from yolo.data.augment import (
    augment_hsv,
    letterbox,
    random_perspective,
    xywhn2xyxy,
    xyxy2xywhn,
)

if TYPE_CHECKING:
    from yolo.data.dataset import YOLODataset


@dataclass
class Sample:
    """A training sample with image and labels.

    Attributes:
        img: BGR image, shape (H, W, 3)
        labels: Labels array, shape (N, 5) as [class_id, x1, y1, x2, y2] in pixels
                or [class_id, x, y, w, h] normalized, depending on pipeline stage
        img_size: Target image size
        original_shape: Original image shape (h, w) before any transforms
    """

    img: np.ndarray
    labels: np.ndarray
    img_size: int
    original_shape: tuple[int, int]


class Transform(ABC):
    """Base class for transforms."""

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        """Apply transform to sample."""
        pass


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Mosaic(Transform):
    """4-image mosaic augmentation with random perspective.

    Matches reference behavior: applies random_perspective at the end with border cropping.
    Requires access to dataset for loading additional images.
    """

    def __init__(
        self,
        dataset: YOLODataset,
        p: float = 1.0,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
    ):
        self.dataset = dataset
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample

        s = sample.img_size
        mosaic_border = (-s // 2, -s // 2)
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)

        indices = [self.dataset.indices[0]] + random.choices(self.dataset.indices, k=3)
        random.shuffle(indices)

        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        labels4 = []

        for i, idx in enumerate(indices):
            img, _, (h, w) = self.dataset._load_image(idx)

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.dataset.labels[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        # Apply random perspective with border (crops to img_size)
        img4, labels4 = random_perspective(
            img4,
            labels4,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            border=mosaic_border,
        )

        return Sample(
            img=img4,
            labels=labels4,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class MixUp(Transform):
    """MixUp augmentation: blend two samples."""

    def __init__(self, dataset: YOLODataset, p: float = 0.0, alpha: float = 32.0):
        self.dataset = dataset
        self.p = p
        self.alpha = alpha

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample

        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, _, (h, w) = self.dataset._load_image(idx2)
        img2, _, _ = letterbox(img2, sample.img_size, auto=False, scaleup=True)

        labels2 = self.dataset.labels[idx2].copy()
        if labels2.size:
            labels2[:, 1:] = xywhn2xyxy(labels2[:, 1:], w, h, 0, 0)

        r = np.random.beta(self.alpha, self.alpha)
        img = (sample.img * r + img2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((sample.labels, labels2), 0)

        return Sample(
            img=img,
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class Letterbox(Transform):
    """Resize and pad image to target size."""

    def __init__(self, scaleup: bool = True):
        self.scaleup = scaleup

    def __call__(self, sample: Sample) -> Sample:
        img, ratio, pad = letterbox(
            sample.img, sample.img_size, auto=False, scaleup=self.scaleup
        )

        labels = sample.labels.copy()
        if labels.size:
            h, w = sample.img.shape[:2]
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1]
            )

        return Sample(
            img=img,
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class RandomPerspective(Transform):
    """Random perspective/affine transformation."""

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def __call__(self, sample: Sample) -> Sample:
        img, labels = random_perspective(
            sample.img,
            sample.labels,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
        )
        return Sample(
            img=img,
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class HSV(Transform):
    """HSV color augmentation."""

    def __init__(self, h: float = 0.015, s: float = 0.7, v: float = 0.4):
        self.h = h
        self.s = s
        self.v = v

    def __call__(self, sample: Sample) -> Sample:
        augment_hsv(sample.img, hgain=self.h, sgain=self.s, vgain=self.v)
        return sample


class RandomFlip(Transform):
    """Random horizontal and/or vertical flip."""

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.0):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, sample: Sample) -> Sample:
        img = sample.img
        labels = sample.labels

        if random.random() < self.p_vertical:
            img = np.flipud(img)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]

        if random.random() < self.p_horizontal:
            img = np.fliplr(img)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]

        return Sample(
            img=np.ascontiguousarray(img),
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class NormalizeLabels(Transform):
    """Convert labels from xyxy pixels to normalized xywh."""

    def __call__(self, sample: Sample) -> Sample:
        labels = sample.labels.copy()
        if len(labels):
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], w=sample.img.shape[1], h=sample.img.shape[0], clip=True, eps=1e-3
            )
        return Sample(
            img=sample.img,
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


class Albumentations(Transform):
    """Wrapper for albumentations transforms.

    Applies image-only transforms from albumentations library. For transforms that
    need bbox awareness, use the bbox_format parameter.

    Default transforms match reference YOLOv9: Blur, MedianBlur, ToGray, CLAHE.
    """

    def __init__(
        self,
        blur_p: float = 0.01,
        median_blur_p: float = 0.01,
        to_gray_p: float = 0.01,
        clahe_p: float = 0.01,
    ):
        self.transform = A.Compose(
            [
                A.Blur(blur_limit=7, p=blur_p),
                A.MedianBlur(blur_limit=3, p=median_blur_p),
                A.ToGray(p=to_gray_p),
                A.CLAHE(p=clahe_p),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def __call__(self, sample: Sample) -> Sample:
        # Labels are in normalized xywh format at this point
        if len(sample.labels):
            bboxes = sample.labels[:, 1:5].tolist()
            class_labels = sample.labels[:, 0].tolist()
        else:
            bboxes = []
            class_labels = []

        result = self.transform(
            image=sample.img,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        img = result["image"]
        if result["bboxes"]:
            labels = np.array(
                [[c, *b] for c, b in zip(result["class_labels"], result["bboxes"])]
            )
        else:
            labels = np.zeros((0, 5))

        return Sample(
            img=img,
            labels=labels,
            img_size=sample.img_size,
            original_shape=sample.original_shape,
        )


def default_train_transforms(
    dataset: YOLODataset,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    flipud: float = 0.0,
    fliplr: float = 0.5,
) -> Compose:
    """Create default training transform pipeline.

    Pipeline order matches reference YOLOv9:
    1. Mosaic (includes random_perspective with border)
    2. MixUp
    3. NormalizeLabels (xyxy -> xywhn)
    4. Albumentations (Blur, MedianBlur, ToGray, CLAHE)
    5. HSV
    6. RandomFlip
    """
    return Compose([
        Mosaic(
            dataset,
            p=mosaic,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
        ),
        MixUp(dataset, p=mixup),
        NormalizeLabels(),
        Albumentations(),
        HSV(h=hsv_h, s=hsv_s, v=hsv_v),
        RandomFlip(p_horizontal=fliplr, p_vertical=flipud),
    ])


def default_val_transforms() -> Compose:
    """Create default validation transform pipeline."""
    return Compose([
        Letterbox(scaleup=False),
        NormalizeLabels(),
    ])
