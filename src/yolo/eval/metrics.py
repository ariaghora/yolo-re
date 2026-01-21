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

    # Collect all predictions and ground truths by class
    all_aps: dict[float, list[float]] = {t: [] for t in iou_thresholds}

    for cls_id in range(num_classes):
        # Gather predictions for this class across all images
        cls_preds: list[tuple[int, float, Tensor]] = []  # (img_id, score, box)
        cls_gts: list[tuple[int, Tensor]] = []  # (img_id, box)
        gt_matched: dict[int, list[bool]] = {}  # Track which GTs are matched

        for img_id in range(num_images):
            # Predictions
            if len(pred_classes[img_id]) > 0:
                mask = pred_classes[img_id] == cls_id
                for score, box in zip(
                    pred_scores[img_id][mask].tolist(),
                    pred_boxes[img_id][mask],
                ):
                    cls_preds.append((img_id, score, box))

            # Ground truths
            if len(gt_classes[img_id]) > 0:
                mask = gt_classes[img_id] == cls_id
                boxes = gt_boxes[img_id][mask]
                gt_matched[img_id] = [False] * len(boxes)
                for box in boxes:
                    cls_gts.append((img_id, box))

        if not cls_gts:
            # No ground truth for this class
            continue

        if not cls_preds:
            # No predictions, AP = 0
            for t in iou_thresholds:
                all_aps[t].append(0.0)
            continue

        # Sort predictions by score descending
        cls_preds.sort(key=lambda x: x[1], reverse=True)
        num_gt = len(cls_gts)

        # Evaluate at each IoU threshold
        for iou_thresh in iou_thresholds:
            # Reset GT matched flags
            matched = {k: [False] * len(v) for k, v in gt_matched.items()}

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))

            debug_tp_count = 0
            debug_fp_count = 0

            for pred_idx, (img_id, score, pred_box) in enumerate(cls_preds):
                # Get GTs for this image
                gt_mask = gt_classes[img_id] == cls_id
                img_gt_boxes = gt_boxes[img_id][gt_mask]

                if len(img_gt_boxes) == 0:
                    fp[pred_idx] = 1
                    debug_fp_count += 1
                    continue

                # Compute IoU with all GTs
                ious = box_iou(pred_box.unsqueeze(0), img_gt_boxes).squeeze(0)
                best_iou, best_gt = ious.max(dim=0)

                if best_iou >= iou_thresh and not matched[img_id][best_gt]:
                    tp[pred_idx] = 1
                    matched[img_id][best_gt] = True
                    debug_tp_count += 1
                else:
                    fp[pred_idx] = 1
                    debug_fp_count += 1

            # Debug: log TP/FP for key classes at IoU 0.5
            if iou_thresh == 0.5 and cls_id in [45, 49, 50] and num_gt > 0:
                import logging
                logger = logging.getLogger(__name__)
                # Count images with GT for this class
                imgs_with_gt = sum(
                    1 for img_id in range(num_images)
                    if len(gt_classes[img_id]) > 0 and (gt_classes[img_id] == cls_id).any()
                )
                # Count images with preds for this class
                imgs_with_pred = len(set(p[0] for p in cls_preds))
                logger.info(
                    f"DEBUG mAP cls {cls_id} @0.5: {len(cls_preds)} preds on {imgs_with_pred} "
                    f"imgs, {num_gt} GTs on {imgs_with_gt} imgs, TP={debug_tp_count}, "
                    f"FP={debug_fp_count}"
                )

            # Compute precision/recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / num_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = compute_ap(recall, precision)
            all_aps[iou_thresh].append(ap)

    # Aggregate results
    results: dict[str, float] = {}

    # mAP at specific thresholds
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
