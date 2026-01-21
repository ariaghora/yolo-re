"""Visualization utilities for debugging."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch import Tensor

# COCO class names for labeling
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def draw_boxes(
    image: np.ndarray,
    boxes: Tensor | np.ndarray,
    classes: Tensor | np.ndarray,
    scores: Tensor | np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label_prefix: str = "",
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: BGR image as numpy array.
        boxes: Bounding boxes in xyxy format, shape (N, 4).
        classes: Class indices, shape (N,).
        scores: Optional confidence scores, shape (N,).
        color: BGR color tuple for boxes.
        thickness: Line thickness.
        label_prefix: Prefix for labels (e.g., "GT:").

    Returns:
        Image with boxes drawn.
    """
    img = image.copy()

    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(classes, "cpu"):
        classes = classes.cpu().numpy()
    if scores is not None and hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label
        cls_id = int(cls)
        name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
        if scores is not None:
            label = f"{label_prefix}{name} {scores[i]:.2f}"
        else:
            label = f"{label_prefix}{name}"

        # Background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img


def save_debug_images(
    images: list[np.ndarray],
    pred_boxes: list[Tensor],
    pred_classes: list[Tensor],
    pred_scores: list[Tensor],
    gt_boxes: list[Tensor],
    gt_classes: list[Tensor],
    output_dir: Path | str,
    epoch: int,
    max_images: int = 10,
) -> None:
    """Save debug images with predictions (green) and GT (red).

    Args:
        images: List of BGR images.
        pred_boxes: List of predicted boxes per image, xyxy format.
        pred_classes: List of predicted classes per image.
        pred_scores: List of prediction scores per image.
        gt_boxes: List of GT boxes per image, xyxy format.
        gt_classes: List of GT classes per image.
        output_dir: Directory to save images.
        epoch: Current epoch number.
        max_images: Maximum number of images to save.
    """
    output_path = Path(output_dir) / f"debug_epoch{epoch}"
    output_path.mkdir(parents=True, exist_ok=True)

    n_images = min(len(images), max_images)
    for i in range(n_images):
        img = images[i].copy()

        # Draw GT boxes in red
        if len(gt_boxes[i]) > 0:
            img = draw_boxes(
                img, gt_boxes[i], gt_classes[i], color=(0, 0, 255), label_prefix="GT:"
            )

        # Draw top predictions in green (limit to avoid clutter)
        if len(pred_boxes[i]) > 0:
            # Take top 20 by score
            n_pred = min(len(pred_boxes[i]), 20)
            img = draw_boxes(
                img,
                pred_boxes[i][:n_pred],
                pred_classes[i][:n_pred],
                pred_scores[i][:n_pred],
                color=(0, 255, 0),
            )

        cv2.imwrite(str(output_path / f"sample_{i}.jpg"), img)
