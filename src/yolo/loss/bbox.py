"""Bounding box regression loss components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yolo.loss.iou import bbox_iou


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh: bool = True) -> Tensor:
    """Transform distance predictions (ltrb) to bounding boxes.

    Args:
        distance: Distance predictions, shape (..., 4) as (left, top, right, bottom)
        anchor_points: Anchor center points, shape (..., 2)
        xywh: If True, return (x_center, y_center, w, h). If False, return (x1, y1, x2, y2).

    Returns:
        Bounding boxes in specified format
    """
    lt, rb = torch.split(distance, 2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=-1)
    return torch.cat((x1y1, x2y2), dim=-1)


def bbox2dist(anchor_points: Tensor, bbox: Tensor, reg_max: int) -> Tensor:
    """Transform bounding boxes (xyxy) to distance predictions (ltrb).

    Args:
        anchor_points: Anchor center points, shape (..., 2)
        bbox: Bounding boxes in xyxy format, shape (..., 4)
        reg_max: Maximum regression value for clamping

    Returns:
        Distance values (left, top, right, bottom), clamped to [0, reg_max - 0.01]
    """
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)


class BboxLoss(nn.Module):
    """Combined CIoU loss and Distribution Focal Loss for box regression."""

    def __init__(self, reg_max: int):
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: float,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute box regression losses.

        Args:
            pred_dist: Predicted distributions, shape (batch, num_anchors, 4 * (reg_max + 1))
            pred_bboxes: Decoded predicted boxes, shape (batch, num_anchors, 4)
            anchor_points: Anchor center points, shape (num_anchors, 2)
            target_bboxes: Target boxes, shape (batch, num_anchors, 4)
            target_scores: Target scores, shape (batch, num_anchors, num_classes)
            target_scores_sum: Sum of target scores for normalization
            fg_mask: Foreground mask, shape (batch, num_anchors)

        Returns:
            loss_iou: CIoU loss
            loss_dfl: Distribution Focal Loss
            iou: IoU values for positive samples
        """
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
        pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
        loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """Distribution Focal Loss.

        Computes cross-entropy loss with soft labels interpolated between
        the two nearest integer bins.
        """
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none"
            ).view(target_left.shape)
            * weight_left
        )
        loss_right = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction="none"
            ).view(target_left.shape)
            * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)
