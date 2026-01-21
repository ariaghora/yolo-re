"""COCO-style evaluator for YOLO models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from yolo.eval.metrics import compute_map
from yolo.utils.nms import non_max_suppression

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """COCO-style evaluation for object detection.

    Example:
        evaluator = Evaluator(model, val_loader, num_classes=80)
        metrics = evaluator.evaluate()
        print(f"mAP@50: {metrics['map50']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int = 80,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        device: str | torch.device = "auto",
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            dataloader: Validation dataloader.
            num_classes: Number of object classes.
            conf_thres: Confidence threshold for NMS (low for mAP eval).
            iou_thres: IoU threshold for NMS.
            device: Device to run on. "auto" detects best available.
        """
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if device == "auto":
            from yolo.utils.device import get_device

            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the full dataset.

        Returns:
            Dictionary with mAP@50, mAP@75, mAP@50:95, val_loss.
        """
        self.model.eval()

        all_pred_boxes: list[Tensor] = []
        all_pred_scores: list[Tensor] = []
        all_pred_classes: list[Tensor] = []
        all_gt_boxes: list[Tensor] = []
        all_gt_classes: list[Tensor] = []

        for batch_idx, (images, targets, _, orig_shapes) in enumerate(self.dataloader):
            images = images.to(self.device, non_blocking=True)
            batch_size = images.shape[0]
            img_size = images.shape[2]  # Assume square

            # Forward pass
            outputs = self.model(images)

            # Get decoded predictions
            if isinstance(outputs, tuple):
                preds = outputs[0]  # (batch, num_anchors, 4 + num_classes)
            else:
                # Raw feature maps - need to decode
                raise NotImplementedError("Raw feature map decoding not implemented")

            # Apply NMS
            detections = non_max_suppression(
                preds,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )

            # Process each image in batch
            for i in range(batch_size):
                det = detections[i]

                # Store predictions
                if len(det) > 0:
                    all_pred_boxes.append(det[:, :4].cpu())
                    all_pred_scores.append(det[:, 4].cpu())
                    all_pred_classes.append(det[:, 5].long().cpu())
                else:
                    all_pred_boxes.append(torch.zeros((0, 4)))
                    all_pred_scores.append(torch.zeros((0,)))
                    all_pred_classes.append(torch.zeros((0,), dtype=torch.long))

                # Extract ground truth for this image
                img_targets = targets[targets[:, 0] == i]
                if len(img_targets) > 0:
                    # Convert from normalized xywh to pixel xyxy
                    gt_cls = img_targets[:, 1].long().cpu()
                    gt_xywh = img_targets[:, 2:6].cpu()

                    # Scale to image size
                    gt_xywh[:, [0, 2]] *= img_size
                    gt_xywh[:, [1, 3]] *= img_size

                    # Convert xywh to xyxy
                    gt_xyxy = torch.zeros_like(gt_xywh)
                    gt_xyxy[:, 0] = gt_xywh[:, 0] - gt_xywh[:, 2] / 2
                    gt_xyxy[:, 1] = gt_xywh[:, 1] - gt_xywh[:, 3] / 2
                    gt_xyxy[:, 2] = gt_xywh[:, 0] + gt_xywh[:, 2] / 2
                    gt_xyxy[:, 3] = gt_xywh[:, 1] + gt_xywh[:, 3] / 2

                    all_gt_boxes.append(gt_xyxy)
                    all_gt_classes.append(gt_cls)
                else:
                    all_gt_boxes.append(torch.zeros((0, 4)))
                    all_gt_classes.append(torch.zeros((0,), dtype=torch.long))

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Evaluated {batch_idx + 1}/{len(self.dataloader)} batches")

        # Compute mAP
        metrics = compute_map(
            pred_boxes=all_pred_boxes,
            pred_scores=all_pred_scores,
            pred_classes=all_pred_classes,
            gt_boxes=all_gt_boxes,
            gt_classes=all_gt_classes,
            num_classes=self.num_classes,
        )

        logger.info(
            f"Evaluation complete: mAP@50={metrics['map50']:.4f}, "
            f"mAP@75={metrics['map75']:.4f}, mAP@50:95={metrics['map']:.4f}"
        )

        return metrics
