"""Task-Aligned Loss for YOLO training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from yolo.loss.assigner import TaskAlignedAssigner
from yolo.loss.bbox import BboxLoss, dist2bbox


@dataclass
class LossConfig:
    """Configuration for TALoss."""

    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    tal_topk: int = 10
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    cls_pw: float = 1.0


def make_anchors(
    feats: list[Tensor], strides: list[int], grid_cell_offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    """Generate anchor points from feature maps.

    Args:
        feats: List of feature maps, each with shape (batch, channels, h, w)
        strides: Stride for each feature map level
        grid_cell_offset: Offset for grid cell centers (0.5 = center of cell)

    Returns:
        anchor_points: Anchor center coordinates, shape (total_anchors, 2)
        stride_tensor: Stride for each anchor, shape (total_anchors, 1)
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def xywh2xyxy(x: Tensor) -> Tensor:
    """Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2)."""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class TALoss(nn.Module):
    """Task-Aligned Loss for YOLO.

    Combines:
    - CIoU loss for box regression
    - BCE loss for classification
    - Distribution Focal Loss for fine-grained localization

    Supports both single-head (DetectDFL) and dual-head (DualDetectDFL) models.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int,
        strides: list[int],
        config: LossConfig | None = None,
    ):
        """Initialize TALoss.

        Args:
            num_classes: Number of object classes
            reg_max: Maximum regression value for DFL
            strides: Feature map strides (e.g., [8, 16, 32])
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.config = config or LossConfig()
        self.no = reg_max * 4 + num_classes

        self.assigner = TaskAlignedAssigner(
            topk=self.config.tal_topk,
            num_classes=num_classes,
            alpha=self.config.tal_alpha,
            beta=self.config.tal_beta,
        )
        self.bbox_loss = BboxLoss(reg_max - 1)
        self.proj = nn.Parameter(torch.arange(reg_max, dtype=torch.float), requires_grad=False)

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.cls_pw]), reduction="none"
        )

    def forward(
        self,
        preds: tuple[Tensor, list[Tensor]] | list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute loss.

        Args:
            preds: Model predictions. Either:
                - list[Tensor]: Raw feature maps from single head
                - tuple[Tensor, list[Tensor]]: (decoded, raw) from DetectDFL
                - tuple[Tensor, tuple[list, list]]: From DualDetectDFL
            targets: Ground truth, shape (N, 6) where each row is
                     (image_idx, class_id, x_center, y_center, w, h) normalized to [0, 1]

        Returns:
            total_loss: Scalar loss value (sum of components * batch_size)
            loss_items: Tensor of [box_loss, cls_loss, dfl_loss] for logging
        """
        if isinstance(preds, tuple):
            raw = preds[1]
            if isinstance(raw, tuple):
                return self._forward_dual((preds[0], raw), targets)
        return self._forward_single(preds, targets)

    def _forward_single(
        self,
        preds: tuple[Tensor, list[Tensor]] | list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        feats = preds[1] if isinstance(preds, tuple) else preds
        device = feats[0].device
        loss = torch.zeros(3, device=device)

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.num_classes), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.strides[0]
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)

        targets_processed = self._preprocess(targets, batch_size, imgsz[[1, 0, 1, 0]], device)
        gt_labels, gt_bboxes = targets_processed.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self._bbox_decode(anchor_points, pred_distri, dtype)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2], _ = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.config.box_gain
        loss[1] *= self.config.cls_gain
        loss[2] *= self.config.dfl_gain

        return loss.sum() * batch_size, loss.detach()

    def _forward_dual(
        self,
        preds: tuple[Tensor, tuple[list[Tensor], list[Tensor]]],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for dual-head models (YOLOv9 with auxiliary head)."""
        feats_aux = preds[1][0]
        feats_main = preds[1][1]
        device = feats_main[0].device
        loss = torch.zeros(3, device=device)

        pred_distri_aux, pred_scores_aux = torch.cat(
            [xi.view(feats_aux[0].shape[0], self.no, -1) for xi in feats_aux], 2
        ).split((self.reg_max * 4, self.num_classes), 1)
        pred_scores_aux = pred_scores_aux.permute(0, 2, 1).contiguous()
        pred_distri_aux = pred_distri_aux.permute(0, 2, 1).contiguous()

        pred_distri_main, pred_scores_main = torch.cat(
            [xi.view(feats_main[0].shape[0], self.no, -1) for xi in feats_main], 2
        ).split((self.reg_max * 4, self.num_classes), 1)
        pred_scores_main = pred_scores_main.permute(0, 2, 1).contiguous()
        pred_distri_main = pred_distri_main.permute(0, 2, 1).contiguous()

        dtype = pred_scores_main.dtype
        batch_size = pred_scores_main.shape[0]
        imgsz = torch.tensor(feats_main[0].shape[2:], device=device, dtype=dtype) * self.strides[0]
        anchor_points, stride_tensor = make_anchors(feats_main, self.strides, 0.5)

        targets_processed = self._preprocess(targets, batch_size, imgsz[[1, 0, 1, 0]], device)
        gt_labels, gt_bboxes = targets_processed.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes_aux = self._bbox_decode(anchor_points, pred_distri_aux, dtype)
        pred_bboxes_main = self._bbox_decode(anchor_points, pred_distri_main, dtype)

        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux = self.assigner(
            pred_scores_aux.detach().sigmoid(),
            (pred_bboxes_aux.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_labels_main, target_bboxes_main, target_scores_main, fg_mask_main = self.assigner(
            pred_scores_main.detach().sigmoid(),
            (pred_bboxes_main.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes_aux /= stride_tensor
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)
        target_bboxes_main /= stride_tensor
        target_scores_sum_main = max(target_scores_main.sum(), 1)

        cls_loss_aux = self.bce(pred_scores_aux, target_scores_aux.to(dtype)).sum()
        cls_loss_main = self.bce(pred_scores_main, target_scores_main.to(dtype)).sum()
        loss[1] = cls_loss_aux / target_scores_sum_aux * 0.25
        loss[1] += cls_loss_main / target_scores_sum_main

        if fg_mask_aux.sum():
            loss0_aux, loss2_aux, _ = self.bbox_loss(
                pred_distri_aux,
                pred_bboxes_aux,
                anchor_points,
                target_bboxes_aux,
                target_scores_aux,
                target_scores_sum_aux,
                fg_mask_aux,
            )
            loss[0] += loss0_aux * 0.25
            loss[2] += loss2_aux * 0.25

        if fg_mask_main.sum():
            loss0_main, loss2_main, _ = self.bbox_loss(
                pred_distri_main,
                pred_bboxes_main,
                anchor_points,
                target_bboxes_main,
                target_scores_main,
                target_scores_sum_main,
                fg_mask_main,
            )
            loss[0] += loss0_main
            loss[2] += loss2_main

        loss[0] *= self.config.box_gain
        loss[1] *= self.config.cls_gain
        loss[2] *= self.config.dfl_gain

        return loss.sum() * batch_size, loss.detach()

    def _preprocess(
        self, targets: Tensor, batch_size: int, scale_tensor: Tensor, device: torch.device
    ) -> Tensor:
        """Preprocess targets into padded batch format.

        Args:
            targets: Raw targets, shape (N, 6) as (img_idx, cls, x, y, w, h)
            batch_size: Number of images in batch
            scale_tensor: Tensor to scale normalized coords to image size
            device: Target device

        Returns:
            Padded targets, shape (batch_size, max_boxes, 5) as (cls, x1, y1, x2, y2)
        """
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max().int(), 5, device=device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor, dtype: torch.dtype) -> Tensor:
        """Decode predicted distributions to bounding boxes."""
        b, a, c = pred_dist.shape
        proj = self.proj.to(device=pred_dist.device, dtype=dtype)
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
