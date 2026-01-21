"""YOLO Trainer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import SGD
from tqdm import tqdm

from yolo.data.config import DataConfig
from yolo.data.dataset import create_dataloader
from yolo.loss.tal import LossConfig, TALoss
from yolo.model.model import YOLO
from yolo.train.config import TrainConfig
from yolo.train.scheduler import WarmupCosineScheduler
from yolo.utils.device import get_device

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """YOLO Trainer.

    Supports multiple construction patterns:

        # From YAML (references model and data configs)
        trainer = Trainer.from_yaml("configs/train/default.yaml")

        # With pre-built model and data config
        trainer = Trainer(model=model, data=data_config)

        # With custom optimizer
        optimizer = AdamW(model.optim_groups(0.0005), lr=0.01)
        trainer = Trainer(model=model, data=data_config, optimizer=optimizer)

        # With custom loss
        trainer = Trainer(model=model, data=data_config, loss_fn=my_loss)

        # Quick override of common params
        trainer = Trainer(model=model, data=data_config, epochs=50, lr=0.001)
    """

    def __init__(
        self,
        model: nn.Module,
        data: DataConfig,
        config: TrainConfig | None = None,
        optimizer: Optimizer | None = None,
        loss_fn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train. Must be YOLO or have optim_groups() method
                if optimizer is not provided.
            data: Data configuration.
            config: Training configuration. If None, uses defaults.
            optimizer: Optional pre-built optimizer. If None, builds SGD
                using model.optim_groups().
            loss_fn: Optional loss function. If None, builds TALoss from
                model's detection head info.
            **kwargs: Override any TrainConfig field (epochs, lr, etc).
        """
        # Build config with overrides
        if config is None:
            config = TrainConfig(**kwargs)  # type: ignore[arg-type]
        elif kwargs:
            config_dict = {k: v for k, v in config.__dict__.items()}
            config_dict.update(kwargs)
            config = TrainConfig(**config_dict)  # type: ignore[arg-type]

        self.config = config
        self.data_config = data
        self.device = get_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Model
        self.model = model
        self.model.to(self.device)

        # Enable cudnn benchmark for faster training with fixed input sizes
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Loss
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            num_classes, reg_max, strides = self._get_detect_info()
            self.loss_fn = TALoss(
                num_classes=num_classes,
                reg_max=reg_max,
                strides=strides,
                config=LossConfig(),
            )
        self.loss_fn.to(self.device)

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if not isinstance(model, YOLO):
                raise TypeError(
                    "Model must be YOLO instance or pass optimizer explicitly"
                )
            self.optimizer = SGD(
                model.optim_groups(config.weight_decay),
                lr=config.lr,
                momentum=config.momentum,
            )

        # Data loaders
        self.train_loader: DataLoader[tuple[Tensor, Tensor, str, tuple[int, int]]] = (
            create_dataloader(data, train=True)
        )
        self.val_loader: DataLoader[tuple[Tensor, Tensor, str, tuple[int, int]]] | None = None
        if data.val_path is not None:
            self.val_loader = create_dataloader(data, train=False)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_epochs=config.epochs,
            steps_per_epoch=len(self.train_loader),
            warmup_epochs=config.warmup_epochs,
            warmup_momentum=config.warmup_momentum,
            warmup_bias_lr=config.warmup_bias_lr,
            lrf=config.lrf,
        )

        # AMP (CUDA only - MPS doesn't support GradScaler)
        self.scaler: GradScaler | None = None
        if config.amp and self.device.type == "cuda":
            self.scaler = GradScaler(self.device.type)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_fitness = 0.0

        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self._count_params():,} parameters")
        logger.info(f"Training: {config.epochs} epochs, batch {data.batch_size}")

    def _get_detect_info(self) -> tuple[int, int, list[int]]:
        """Get detection head info: num_classes, reg_max, strides."""
        from yolo.heads.detect import DetectDFL, DualDetectDFL

        detect = None
        if isinstance(self.model, YOLO):
            for layer in self.model.layers.values():
                if isinstance(layer, (DetectDFL, DualDetectDFL)):
                    detect = layer
                    break

        if detect is None:
            for module in self.model.modules():
                if isinstance(module, (DetectDFL, DualDetectDFL)):
                    detect = module
                    break

        if detect is None:
            raise ValueError("Could not find detection head in model")

        strides = detect.stride.int().tolist()
        return detect.num_classes, detect.reg_max, strides

    def _count_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self) -> None:
        """Run full training loop."""
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            self.train_one_epoch()

            # Validation at specified interval
            should_validate = (
                self.val_loader is not None
                and self.config.val_period > 0
                and (epoch + 1) % self.config.val_period == 0
            )

            if should_validate:
                metrics = self.validate()
                fitness = metrics.get("map50", 0.0)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.save_checkpoint("best.pt")

            if self.config.save_period > 0 and (epoch + 1) % self.config.save_period == 0:
                self.save_checkpoint(f"epoch{epoch + 1}.pt")

        # Final validation if not done on last epoch
        if self.val_loader is not None and self.config.val_period > 0:
            last_val_epoch = (self.config.epochs // self.config.val_period) * self.config.val_period
            if last_val_epoch != self.config.epochs:
                self.validate()

        self.save_checkpoint("last.pt")
        logger.info("Training complete")

    def train_one_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_box = 0.0
        total_cls = 0.0
        total_dfl = 0.0
        num_batches = 0

        start = time.perf_counter()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        for batch_idx, (images, targets, _, _) in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast(self.device.type):
                    outputs = self.model(images)
                    loss, loss_items = self._compute_loss(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, loss_items = self._compute_loss(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            box_val = loss_items[0].item()
            cls_val = loss_items[1].item()
            dfl_val = loss_items[2].item()
            total_box += box_val
            total_cls += cls_val
            total_dfl += dfl_val
            num_batches += 1

            # Update progress bar with current losses
            pbar.set_postfix(box=f"{box_val:.3f}", cls=f"{cls_val:.3f}", dfl=f"{dfl_val:.3f}")

        elapsed = time.perf_counter() - start
        n = max(num_batches, 1)
        avg_box, avg_cls, avg_dfl = total_box / n, total_cls / n, total_dfl / n

        tqdm.write(
            f"Epoch {self.epoch + 1} done in {elapsed:.1f}s - "
            f"box: {avg_box:.4f} cls: {avg_cls:.4f} dfl: {avg_dfl:.4f}"
        )

        return {
            "box_loss": avg_box,
            "cls_loss": avg_cls,
            "dfl_loss": avg_dfl,
        }

    def _compute_loss(
        self,
        outputs: Tensor | tuple[Tensor, list[Tensor]],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute loss from model outputs.

        Returns:
            Tuple of (total_loss, loss_items) where loss_items is (box, cls, dfl).
        """
        if isinstance(outputs, tuple):
            preds, aux_preds = outputs
            return self.loss_fn(preds, targets, aux_preds)
        return self.loss_fn(outputs, targets)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation with mAP computation."""
        if self.val_loader is None:
            return {}

        from yolo.eval.evaluator import Evaluator

        # Get num_classes from model
        num_classes, _, _ = self._get_detect_info()

        evaluator = Evaluator(
            model=self.model,
            dataloader=self.val_loader,
            num_classes=num_classes,
            device=self.device,
            debug_dir=self.output_dir / "debug",
        )

        metrics = evaluator.evaluate(epoch=self.epoch + 1)
        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_fitness": self.best_fitness,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint and resume training."""
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        logger.info(f"Resumed from {path} at epoch {self.epoch}")
