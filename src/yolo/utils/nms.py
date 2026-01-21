"""Non-Maximum Suppression for YOLO detections."""

from __future__ import annotations

import torch
from torch import Tensor


def non_max_suppression(
    predictions: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    classes: list[int] | None = None,
    agnostic: bool = False,
) -> list[Tensor]:
    """Apply Non-Maximum Suppression to model predictions.

    Args:
        predictions: Raw model output, shape (batch, num_anchors, 4 + num_classes).
            Boxes are in xyxy format, class scores are raw (pre-sigmoid for cls).
        conf_thres: Confidence threshold for filtering.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        classes: Filter by class indices. None means all classes.
        agnostic: If True, NMS is class-agnostic.

    Returns:
        List of detections per image, each (n, 6) as [x1, y1, x2, y2, conf, cls].
    """
    batch_size = predictions.shape[0]

    # Output list
    output: list[Tensor] = []

    for i in range(batch_size):
        pred = predictions[i]  # (num_anchors, 4 + num_classes)

        # Get boxes and class scores
        boxes = pred[:, :4]  # (num_anchors, 4)
        cls_scores = pred[:, 4:].sigmoid()  # (num_anchors, num_classes)

        # Get max class score and class index per box
        conf, cls_idx = cls_scores.max(dim=1)  # (num_anchors,)

        # Filter by confidence
        mask = conf > conf_thres
        if classes is not None:
            mask &= torch.isin(cls_idx, torch.tensor(classes, device=pred.device))

        boxes = boxes[mask]
        conf = conf[mask]
        cls_idx = cls_idx[mask]

        if boxes.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=pred.device, dtype=pred.dtype))
            continue

        # Apply NMS
        if agnostic:
            # Class-agnostic NMS
            keep = _nms(boxes, conf, iou_thres)
        else:
            # Per-class NMS using class offset trick
            # Offset boxes by class so different classes don't suppress each other
            max_coord = boxes.max()
            offsets = cls_idx.float() * (max_coord + 1)
            boxes_offset = boxes + offsets[:, None]
            keep = _nms(boxes_offset, conf, iou_thres)

        # Limit detections
        keep = keep[:max_det]

        # Build output: [x1, y1, x2, y2, conf, cls]
        det = torch.cat(
            [boxes[keep], conf[keep, None], cls_idx[keep, None].float()],
            dim=1,
        )
        output.append(det)

    return output


def _nms(boxes: Tensor, scores: Tensor, iou_thres: float) -> Tensor:
    """Apply NMS using torchvision if available, else pure PyTorch."""
    try:
        from torchvision.ops import nms

        return nms(boxes, scores, iou_thres)
    except ImportError:
        return _nms_pure(boxes, scores, iou_thres)


def _nms_pure(boxes: Tensor, scores: Tensor, iou_thres: float) -> Tensor:
    """Pure PyTorch NMS implementation."""
    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = _box_iou(boxes[idx : idx + 1], boxes[remaining]).squeeze(0)

        # Keep boxes with IoU below threshold
        mask = ious <= iou_thres
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        IoU matrix (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # IoU
    union = area1[:, None] + area2 - inter
    return inter / union
