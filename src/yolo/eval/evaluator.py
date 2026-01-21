"""COCO-style evaluator for YOLO models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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
        debug_dir: str | Path | None = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            dataloader: Validation dataloader.
            num_classes: Number of object classes.
            conf_thres: Confidence threshold for NMS (low for mAP eval).
            iou_thres: IoU threshold for NMS.
            device: Device to run on. "auto" detects best available.
            debug_dir: Directory to save debug visualizations. None to disable.
        """
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug_dir = Path(debug_dir) if debug_dir else None

        if device == "auto":
            from yolo.utils.device import get_device

            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> dict[str, float]:
        """Run evaluation on the full dataset.

        Args:
            epoch: Current epoch number (for debug image naming).

        Returns:
            Dictionary with mAP@50, mAP@75, mAP@50:95, val_loss.
        """
        self.model.eval()

        all_pred_boxes: list[Tensor] = []
        all_pred_scores: list[Tensor] = []
        all_pred_classes: list[Tensor] = []
        all_gt_boxes: list[Tensor] = []
        all_gt_classes: list[Tensor] = []

        # For debug visualization
        debug_images: list[np.ndarray] = []
        debug_pred_boxes: list[Tensor] = []
        debug_pred_scores: list[Tensor] = []
        debug_pred_classes: list[Tensor] = []
        debug_gt_boxes: list[Tensor] = []
        debug_gt_classes: list[Tensor] = []
        max_debug_images = 10

        for batch_idx, (images, targets, _, orig_shapes) in enumerate(self.dataloader):
            images = images.to(self.device, non_blocking=True)
            batch_size = images.shape[0]
            img_size = images.shape[2]  # Assume square

            # Forward pass
            outputs = self.model(images)

            # Get decoded predictions
            if isinstance(outputs, tuple):
                preds = outputs[0]
                # DualDetectDFL returns [aux_decoded, main_decoded] - use main branch
                if isinstance(preds, list):
                    preds = preds[1]  # main branch
                # preds shape: (batch, 4 + num_classes, num_anchors)
            else:
                # Raw feature maps - need to decode
                raise NotImplementedError("Raw feature map decoding not implemented")

            # Transpose to (batch, num_anchors, 4 + num_classes) for NMS
            preds = preds.permute(0, 2, 1).contiguous()

            # Debug: log prediction stats on first batch
            if batch_idx == 0:
                boxes = preds[0, :, :4]
                cls_scores = preds[0, :, 4:]
                logger.info(f"DEBUG pred boxes range: min={boxes.min():.1f}, max={boxes.max():.1f}")
                min_s, max_s = cls_scores.min(), cls_scores.max()
                logger.info(f"DEBUG pred cls scores: min={min_s:.4f}, max={max_s:.4f}")
                logger.info(f"DEBUG cls scores > 0.001: {(cls_scores > 0.001).sum().item()}")
                logger.info(f"DEBUG cls scores > 0.1: {(cls_scores > 0.1).sum().item()}")
                logger.info(f"DEBUG cls scores > 0.25: {(cls_scores > 0.25).sum().item()}")

            # Apply NMS
            detections = non_max_suppression(
                preds,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )

            # Debug: log detection counts on first batch
            if batch_idx == 0:
                logger.info(f"DEBUG detections per image: {[len(d) for d in detections]}")

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

                    # Debug: log GT info and best IoU on first image
                    if batch_idx == 0 and i == 0:
                        logger.info(f"DEBUG GT boxes: {len(gt_xyxy)}")
                        logger.info(f"DEBUG GT classes: {gt_cls.tolist()[:5]}...")
                        logger.info(f"DEBUG GT box[0] xyxy: {gt_xyxy[0].tolist()}")
                        if len(det) > 0:
                            logger.info(f"DEBUG pred box[0] xyxy: {det[0, :4].tolist()}")
                            conf, cls = det[0, 4], int(det[0, 5])
                            logger.info(f"DEBUG pred box[0] conf/cls: {conf:.4f}, {cls}")
                            # Compute best IoU between any pred and any GT
                            from yolo.eval.metrics import box_iou
                            ious = box_iou(det[:, :4].cpu(), gt_xyxy)
                            best_iou = ious.max().item()
                            logger.info(f"DEBUG best pred-GT IoU: {best_iou:.4f}")

                    all_gt_boxes.append(gt_xyxy)
                    all_gt_classes.append(gt_cls)
                else:
                    all_gt_boxes.append(torch.zeros((0, 4)))
                    all_gt_classes.append(torch.zeros((0,), dtype=torch.long))

                # Collect debug images (only first N with GT)
                if self.debug_dir and len(debug_images) < max_debug_images:
                    if len(img_targets) > 0:  # Only save images that have GT
                        # Convert image tensor to numpy BGR
                        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                        img_np = (img_np * 255).astype(np.uint8)
                        img_np = img_np[:, :, ::-1]  # RGB to BGR
                        debug_images.append(img_np.copy())
                        debug_pred_boxes.append(all_pred_boxes[-1])
                        debug_pred_scores.append(all_pred_scores[-1])
                        debug_pred_classes.append(all_pred_classes[-1])
                        debug_gt_boxes.append(all_gt_boxes[-1])
                        debug_gt_classes.append(all_gt_classes[-1])

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Evaluated {batch_idx + 1}/{len(self.dataloader)} batches")

        # Save debug visualizations
        if self.debug_dir and debug_images:
            from yolo.utils.visualize import save_debug_images
            save_debug_images(
                images=debug_images,
                pred_boxes=debug_pred_boxes,
                pred_classes=debug_pred_classes,
                pred_scores=debug_pred_scores,
                gt_boxes=debug_gt_boxes,
                gt_classes=debug_gt_classes,
                output_dir=self.debug_dir,
                epoch=epoch,
                max_images=max_debug_images,
            )
            logger.info(
                f"Saved {len(debug_images)} debug images to {self.debug_dir}/debug_epoch{epoch}"
            )

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
