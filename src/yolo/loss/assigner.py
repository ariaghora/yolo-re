"""Task-Aligned Assigner for matching predictions to ground truth."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yolo.loss.iou import bbox_iou


def select_candidates_in_gts(xy_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9) -> Tensor:
    """Select anchor centers that fall inside ground truth boxes.

    Args:
        xy_centers: Anchor center points, shape (num_anchors, 2)
        gt_bboxes: Ground truth boxes in xyxy format, shape (batch, n_boxes, 4)
        eps: Small value for numerical stability

    Returns:
        Boolean mask, shape (batch, n_boxes, num_anchors)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
        bs, n_boxes, n_anchors, -1
    )
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(
    mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Resolve cases where one anchor is assigned to multiple GT boxes.

    Args:
        mask_pos: Positive mask, shape (batch, n_max_boxes, num_anchors)
        overlaps: IoU overlaps, shape (batch, n_max_boxes, num_anchors)
        n_max_boxes: Maximum number of boxes

    Returns:
        target_gt_idx: Index of assigned GT for each anchor, shape (batch, num_anchors)
        fg_mask: Foreground mask, shape (batch, num_anchors)
        mask_pos: Updated positive mask
    """
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner for YOLO.

    Assigns ground truth boxes to anchors based on a combination of
    classification score and IoU overlap.

    Reference: https://arxiv.org/abs/2108.07755
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground truth to predictions.

        Args:
            pd_scores: Predicted scores (after sigmoid), shape (batch, num_anchors, num_classes)
            pd_bboxes: Predicted boxes in xyxy, shape (batch, num_anchors, 4)
            anc_points: Anchor center points, shape (num_anchors, 2)
            gt_labels: Ground truth labels, shape (batch, n_max_boxes, 1)
            gt_bboxes: Ground truth boxes in xyxy, shape (batch, n_max_boxes, 4)
            mask_gt: Valid GT mask, shape (batch, n_max_boxes, 1)

        Returns:
            target_labels: Assigned labels, shape (batch, num_anchors)
            target_bboxes: Assigned boxes, shape (batch, num_anchors, 4)
            target_scores: Soft target scores, shape (batch, num_anchors, num_classes)
            fg_mask: Foreground mask, shape (batch, num_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def _get_pos_mask(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        anc_points: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        align_metric, overlaps = self._get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        mask_topk = self._select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(
        self, pd_scores: Tensor, pd_bboxes: Tensor, gt_labels: Tensor, gt_bboxes: Tensor
    ) -> tuple[Tensor, Tensor]:
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros(
            [2, self.bs, self.n_max_boxes], dtype=torch.long, device=pd_scores.device
        )
        ind[0] = (
            torch.arange(end=self.bs, device=pd_scores.device)
            .view(-1, 1)
            .repeat(1, self.n_max_boxes)
        )
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], :, ind[1]]

        overlaps = (
            bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True)
            .squeeze(3)
            .clamp(0)
        )
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def _select_topk_candidates(
        self, metrics: Tensor, largest: bool = True, topk_mask: Tensor | None = None
    ) -> Tensor:
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True).values > self.eps).tile(
                [1, 1, self.topk]
            )
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def _get_targets(
        self, gt_labels: Tensor, gt_bboxes: Tensor, target_gt_idx: Tensor, fg_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
