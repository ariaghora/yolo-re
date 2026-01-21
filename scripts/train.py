#!/usr/bin/env python3
"""Train YOLO on a dataset.

Usage:
    python scripts/train.py --data /path/to/images/train --epochs 100
    python scripts/train.py --data /path/to/dataset --weights pretrained.pt --lr 0.001
    python scripts/train.py --data /path/to/dataset --config configs/models/gelan-c.yaml
"""

from __future__ import annotations

import argparse

import torch

from yolo import YOLO, Trainer
from yolo.data.config import AugmentConfig, DataConfig
from yolo.train.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train YOLO")
    parser.add_argument("--data", type=str, required=True, help="Path to training images")
    parser.add_argument("--val", type=str, help="Path to validation images (default: same as data)")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--config", type=str, default="configs/models/gelan-c.yaml",
                        help="Model config")
    parser.add_argument("--weights", type=str, help="Pretrained weights to load")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--augment", type=str, default="full", choices=["full", "light", "minimal"],
                        help="Augmentation preset")
    parser.add_argument("--val-period", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--output", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    args = parser.parse_args()

    model = YOLO.from_yaml(args.config)
    if args.weights:
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    data = DataConfig(
        train_path=args.data,
        val_path=args.val or args.data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        workers=args.workers,
        augment=AugmentConfig.from_preset(args.augment),
    )

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        val_period=args.val_period,
        output_dir=args.output,
        device=args.device,
    )

    trainer = Trainer(model, data, config)
    trainer.train()

    print(f"Training complete. Best mAP@50: {trainer.best_fitness:.4f}")
    print(f"Checkpoints saved to: {args.output}")


if __name__ == "__main__":
    main()
