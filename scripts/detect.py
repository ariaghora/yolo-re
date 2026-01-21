#!/usr/bin/env python3
"""Run YOLO inference on images.

Usage:
    python scripts/detect.py --weights runs/train/best.pt --source image.jpg
    python scripts/detect.py --weights runs/train/best.pt --source images/
    python scripts/detect.py --config configs/models/gelan-c.yaml --source image.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from yolo import YOLO
from yolo.utils.nms import non_max_suppression

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Resize and pad image to square shape.

    Args:
        img: Input image (H, W, C).
        new_shape: Target size.
        color: Padding color.

    Returns:
        Resized image, scale ratios (w, h), padding (w, h).
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)

    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]

    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (int(dw), int(dh))


def scale_boxes(
    boxes: torch.Tensor,
    img_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Scale boxes from model input size to original image size.

    Args:
        boxes: Boxes in xyxy format (N, 4).
        img_shape: Model input shape (h, w).
        orig_shape: Original image shape (h, w).
        ratio_pad: Optional (ratio, pad) from letterbox.

    Returns:
        Scaled boxes.
    """
    if ratio_pad is None:
        gain = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
        pad = (
            (img_shape[1] - orig_shape[1] * gain) / 2,
            (img_shape[0] - orig_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain

    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_shape[0])

    return boxes


def draw_detections(
    img: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    class_names: list[str],
) -> np.ndarray:
    """Draw detection boxes on image.

    Args:
        img: Input image (H, W, C) BGR.
        boxes: Detection boxes (N, 4) xyxy.
        scores: Confidence scores (N,).
        classes: Class indices (N,).
        class_names: List of class names.

    Returns:
        Image with drawn detections.
    """
    img = img.copy()

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(cls_id.item())
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="YOLO inference")
    parser.add_argument("--weights", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, help="Path to model config .yaml file")
    parser.add_argument("--source", type=str, required=True, help="Image or directory")
    parser.add_argument("--img-size", type=int, default=640, help="Inference size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--show", action="store_true", help="Show results")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        from yolo.utils.device import get_device

        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        if "model_state_dict" in ckpt:
            # Need config to build model
            if not args.config:
                raise ValueError("--config required when loading from checkpoint")
            model = YOLO.from_yaml(args.config)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Full model saved
            model = ckpt
    elif args.config:
        model = YOLO.from_yaml(args.config)
    else:
        raise ValueError("Either --weights or --config required")

    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Get input images
    source = Path(args.source)
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = list(source.glob("*.jpg")) + list(source.glob("*.png"))
    else:
        raise ValueError(f"Invalid source: {source}")
    print(f"Found {len(images)} images")

    # Output directory
    output_dir = Path(args.output)
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    total_time = 0.0

    for img_path in images:
        # Load image
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"Failed to load: {img_path}")
            continue

        orig_shape = img0.shape[:2]  # (h, w)

        # Preprocess
        img, ratio_pad, _ = letterbox(img0, args.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(device) / 255.0
        img = img.unsqueeze(0)  # Add batch dim

        # Inference
        t0 = time.time()
        with torch.no_grad():
            outputs = model(img)
        t1 = time.time()
        total_time += t1 - t0

        # Get predictions
        if isinstance(outputs, tuple):
            preds = outputs[0]
        else:
            raise NotImplementedError("Raw output not supported")

        # NMS
        detections = non_max_suppression(
            preds,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )[0]

        # Scale boxes to original size
        if len(detections) > 0:
            detections[:, :4] = scale_boxes(
                detections[:, :4],
                img.shape[2:],
                orig_shape,
                (ratio_pad, (int(ratio_pad[1][0]), int(ratio_pad[1][1]))),
            )

        # Results
        num_det = len(detections)
        print(f"{img_path.name}: {num_det} detections in {(t1 - t0) * 1000:.1f}ms")

        if num_det > 0:
            # Draw and optionally save/show
            result_img = draw_detections(
                img0,
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                COCO_NAMES,
            )

            if args.save:
                out_path = output_dir / img_path.name
                cv2.imwrite(str(out_path), result_img)
                print(f"  Saved to {out_path}")

            if args.show:
                cv2.imshow("Detection", result_img)
                cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    avg_time = total_time / len(images) * 1000 if images else 0
    print(f"\nAverage inference time: {avg_time:.1f}ms per image")


if __name__ == "__main__":
    main()
