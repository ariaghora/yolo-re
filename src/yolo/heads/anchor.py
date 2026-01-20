"""Anchor-free detection utilities.

Reference: _reference/yolov9/utils/tal/anchor_generator.py
"""

import torch
from torch import Tensor


def make_anchors(
    features: list[Tensor], strides: Tensor, grid_cell_offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    """Generate anchor points from feature maps.

    Args:
        features: List of feature tensors, each of shape [B, C, H, W].
        strides: Tensor of stride values for each feature level.
        grid_cell_offset: Offset for grid cell centers. Default 0.5.

    Returns:
        anchor_points: Tensor of shape [total_anchors, 2] with (x, y) coordinates.
        stride_tensor: Tensor of shape [total_anchors, 1] with stride for each anchor.

    Reference: _reference/yolov9/utils/tal/anchor_generator.py::make_anchors
    """
    anchor_points = []
    stride_tensor = []
    dtype, device = features[0].dtype, features[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = features[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride.item(), dtype=dtype, device=device)
        )

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh: bool = True, dim: int = -1) -> Tensor:
    """Transform distance (ltrb) to bounding box (xywh or xyxy).

    Args:
        distance: Distance tensor with left, top, right, bottom values.
        anchor_points: Anchor point coordinates.
        xywh: If True, return xywh format. If False, return xyxy format.
        dim: Dimension to split/concat along.

    Returns:
        Bounding boxes in specified format.

    Reference: _reference/yolov9/utils/tal/anchor_generator.py::dist2bbox
    """
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        center = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((center, wh), dim)
    return torch.cat((x1y1, x2y2), dim)
