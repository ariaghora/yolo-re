"""Image augmentations for YOLO training."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np


def augment_hsv(
    img: np.ndarray,
    hgain: float = 0.5,
    sgain: float = 0.5,
    vgain: float = 0.5,
) -> None:
    """HSV color-space augmentation (in-place).

    Args:
        img: BGR image, shape (H, W, 3)
        hgain: Hue gain factor
        sgain: Saturation gain factor
        vgain: Value gain factor
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints.

    Args:
        img: Input image
        new_shape: Target size (single int for square, or (h, w))
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill
        scaleup: Allow scaling up
        stride: Stride constraint

    Returns:
        img: Resized and padded image
        ratio: (width_ratio, height_ratio)
        pad: (width_pad, height_pad)
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


def random_perspective(
    img: np.ndarray,
    labels: np.ndarray,
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
    shear: float = 10,
    perspective: float = 0.0,
    border: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random perspective transformation.

    Args:
        img: Input image
        labels: Labels in xyxy format, shape (N, 5) as [class, x1, y1, x2, y2]
        degrees: Rotation range
        translate: Translation range (fraction of image size)
        scale: Scale range
        shear: Shear range (degrees)
        perspective: Perspective range
        border: Border size for mosaic

    Returns:
        img: Transformed image
        labels: Transformed labels
    """
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined transformation
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform labels
    n = len(labels)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # Filter candidates
        i = _box_candidates(
            box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.01 if perspective else 0.10
        )
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return img, labels


def _box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2,
    ar_thr: float = 100,
    area_thr: float = 0.1,
    eps: float = 1e-16,
) -> np.ndarray:
    """Filter box candidates by constraints.

    Args:
        box1: Original boxes (4, n) as [x1, y1, x2, y2]
        box2: Augmented boxes (4, n)
        wh_thr: Width/height threshold
        ar_thr: Aspect ratio threshold
        area_thr: Area ratio threshold
        eps: Small value for numerical stability

    Returns:
        Boolean mask of valid boxes
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def xyxy2xywhn(
    x: np.ndarray, w: float = 640, h: float = 640, clip: bool = False, eps: float = 0.0
) -> np.ndarray:
    """Convert boxes from xyxy to normalized xywh.

    Args:
        x: Boxes in xyxy format, shape (N, 4)
        w: Image width
        h: Image height
        clip: Clip to [0, 1]
        eps: Epsilon for clipping

    Returns:
        Boxes in normalized xywh format
    """
    if clip:
        x = x.copy()
        x[:, [0, 2]] = x[:, [0, 2]].clip(eps, w - eps)
        x[:, [1, 3]] = x[:, [1, 3]].clip(eps, h - eps)
    y = np.empty_like(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def xywhn2xyxy(
    x: np.ndarray, w: float = 640, h: float = 640, padw: float = 0, padh: float = 0
) -> np.ndarray:
    """Convert boxes from normalized xywh to xyxy.

    Args:
        x: Boxes in normalized xywh format, shape (N, 4)
        w: Image width
        h: Image height
        padw: Width padding offset
        padh: Height padding offset

    Returns:
        Boxes in xyxy format
    """
    y = np.empty_like(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y
