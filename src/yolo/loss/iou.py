"""IoU (Intersection over Union) calculations for bounding boxes."""

from __future__ import annotations

import math
from enum import Enum

import torch
from torch import Tensor


class IoUType(Enum):
    """IoU calculation type."""

    STANDARD = "iou"
    GIOU = "giou"
    DIOU = "diou"
    CIOU = "ciou"


def bbox_iou(
    box1: Tensor,
    box2: Tensor,
    xywh: bool = False,
    iou_type: IoUType = IoUType.STANDARD,
    eps: float = 1e-7,
) -> Tensor:
    """Calculate IoU between two sets of boxes.

    Args:
        box1: First set of boxes, shape (..., 4)
        box2: Second set of boxes, shape (..., 4)
        xywh: If True, boxes are in (x_center, y_center, w, h) format.
              If False, boxes are in (x1, y1, x2, y2) format.
        iou_type: Type of IoU to compute (STANDARD, GIOU, DIOU, or CIOU)
        eps: Small value to avoid division by zero

    Returns:
        IoU values with same batch dimensions as inputs
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if iou_type in (IoUType.CIOU, IoUType.DIOU, IoUType.GIOU):
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if iou_type in (IoUType.CIOU, IoUType.DIOU):
            c2 = cw**2 + ch**2 + eps
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4
            if iou_type == IoUType.CIOU:
                v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    return iou
