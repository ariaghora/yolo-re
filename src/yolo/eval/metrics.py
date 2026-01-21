"""mAP and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        IoU matrix (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO style).

    Args:
        recall: Recall values, sorted ascending.
        precision: Precision values corresponding to recall.

    Returns:
        Average precision value.
    """
    # Prepend/append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_thresholds)

    for i, t in enumerate(recall_thresholds):
        # Find precision at recall >= t
        idx = np.where(mrec >= t)[0]
        if len(idx) > 0:
            precision_interp[i] = mpre[idx[0]]

    return float(precision_interp.mean())


def compute_map(
    pred_boxes: list[Tensor],
    pred_scores: list[Tensor],
    pred_classes: list[Tensor],
    gt_boxes: list[Tensor],
    gt_classes: list[Tensor],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute mAP at various IoU thresholds.

    Args:
        pred_boxes: List of predicted boxes per image, each (N, 4) xyxy.
        pred_scores: List of confidence scores per image, each (N,).
        pred_classes: List of predicted classes per image, each (N,).
        gt_boxes: List of ground truth boxes per image, each (M, 4) xyxy.
        gt_classes: List of ground truth classes per image, each (M,).
        num_classes: Total number of classes.
        iou_thresholds: IoU thresholds for evaluation. Default is COCO style.

    Returns:
        Dictionary with mAP@50, mAP@75, mAP@50:95, and per-class APs.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5:0.95

    num_images = len(pred_boxes)

    # Precompute IoUs per image and cache GT info per class
    # This avoids redundant computation across IoU thresholds
    
    all_aps: dict[float, list[float]] = {t: [] for t in iou_thresholds}

    for cls_id in range(num_classes):
        # Gather all predictions and GTs for this class
        # Group by image for efficient IoU computation
        img_preds: dict[int, list[tuple[float, int]]] = {}  # img_id -> [(score, local_idx)]
        img_pred_boxes: dict[int, Tensor] = {}  # img_id -> stacked pred boxes
        img_gt_boxes: dict[int, Tensor] = {}  # img_id -> stacked gt boxes
        
        total_gt = 0
        
        for img_id in range(num_images):
            # Predictions for this class
            if len(pred_classes[img_id]) > 0:
                mask = pred_classes[img_id] == cls_id
                if mask.any():
                    scores = pred_scores[img_id][mask]
                    boxes = pred_boxes[img_id][mask]
                    img_pred_boxes[img_id] = boxes
                    img_preds[img_id] = [(s.item(), i) for i, s in enumerate(scores)]
            
            # Ground truths for this class
            if len(gt_classes[img_id]) > 0:
                mask = gt_classes[img_id] == cls_id
                if mask.any():
                    img_gt_boxes[img_id] = gt_boxes[img_id][mask]
                    total_gt += mask.sum().item()

        if total_gt == 0:
            continue

        if not img_preds:
            for t in iou_thresholds:
                all_aps[t].append(0.0)
            continue

        # Precompute IoU matrices for all images that have both preds and GTs
        img_ious: dict[int, Tensor] = {}
        for img_id in img_preds:
            if img_id in img_gt_boxes:
                img_ious[img_id] = box_iou(img_pred_boxes[img_id], img_gt_boxes[img_id])

        # Flatten and sort all predictions by score
        all_preds: list[tuple[float, int, int]] = []  # (score, img_id, local_idx)
        for img_id, preds in img_preds.items():
            for score, local_idx in preds:
                all_preds.append((score, img_id, local_idx))
        all_preds.sort(key=lambda x: x[0], reverse=True)

        num_preds = len(all_preds)

        # Evaluate at each IoU threshold
        for iou_thresh in iou_thresholds:
            # Track matched GTs per image
            matched: dict[int, list[bool]] = {
                img_id: [False] * len(boxes) for img_id, boxes in img_gt_boxes.items()
            }

            tp = np.zeros(num_preds)
            fp = np.zeros(num_preds)

            for pred_idx, (score, img_id, local_idx) in enumerate(all_preds):
                if img_id not in img_gt_boxes:
                    fp[pred_idx] = 1
                    continue

                ious = img_ious[img_id][local_idx]
                best_iou, best_gt_idx = ious.max(dim=0)
                best_gt = int(best_gt_idx.item())

                if best_iou >= iou_thresh and not matched[img_id][best_gt]:
                    tp[pred_idx] = 1
                    matched[img_id][best_gt] = True
                else:
                    fp[pred_idx] = 1

            # Compute precision/recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / total_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = compute_ap(recall, precision)
            all_aps[iou_thresh].append(ap)

    # Aggregate results
    results: dict[str, float] = {}

    if 0.5 in all_aps and all_aps[0.5]:
        results["map50"] = float(np.mean(all_aps[0.5]))
    else:
        results["map50"] = 0.0

    if 0.75 in all_aps and all_aps[0.75]:
        results["map75"] = float(np.mean(all_aps[0.75]))
    else:
        results["map75"] = 0.0

    # mAP@50:95
    all_ap_values = []
    for t in iou_thresholds:
        all_ap_values.extend(all_aps.get(t, []))
    results["map"] = float(np.mean(all_ap_values)) if all_ap_values else 0.0

    return results
