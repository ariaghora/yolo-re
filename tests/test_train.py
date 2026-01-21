"""Tests for training utilities."""

import math

import torch
import torch.nn as nn
from torch.optim import SGD

from yolo.train.config import TrainConfig
from yolo.train.scheduler import WarmupCosineScheduler, one_cycle_lr


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, bias=True)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = TrainConfig()
        assert config.epochs == 100
        assert config.lr == 0.01
        assert config.momentum == 0.937
        assert config.weight_decay == 0.0005
        assert config.warmup_epochs == 3.0
        assert config.lrf == 0.01
        assert config.amp is True

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = TrainConfig(epochs=50, lr=0.001)
        assert config.epochs == 50
        assert config.lr == 0.001


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler."""

    def test_warmup_phase(self) -> None:
        """Test learning rate during warmup."""
        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=10,
            steps_per_epoch=100,
            warmup_epochs=1.0,
            lrf=0.01,
        )

        # At start, lr should be 0.01 (initial)
        assert scheduler.get_lr()[0] == 0.01

        # After half warmup
        for _ in range(50):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        assert 0.004 < lr < 0.006  # ~50% of target

        # After full warmup
        for _ in range(50):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        assert 0.009 < lr < 0.011  # ~100% of target

    def test_cosine_decay(self) -> None:
        """Test cosine decay after warmup."""
        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=10,
            steps_per_epoch=100,
            warmup_epochs=0.0,
            lrf=0.01,
        )

        lr_start = scheduler.get_lr()[0]

        for _ in range(500):
            scheduler.step()
        lr_mid = scheduler.get_lr()[0]

        for _ in range(500):
            scheduler.step()
        lr_end = scheduler.get_lr()[0]

        assert lr_start > lr_mid > lr_end
        assert 0.0 < lr_end < 0.002

    def test_epoch_property(self) -> None:
        """Test epoch calculation."""
        model = DummyModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=10,
            steps_per_epoch=100,
            warmup_epochs=1.0,
            lrf=0.01,
        )

        assert scheduler.epoch == 0.0

        for _ in range(100):
            scheduler.step()
        assert scheduler.epoch == 1.0

        for _ in range(50):
            scheduler.step()
        assert scheduler.epoch == 1.5


class TestOneCycleLr:
    """Tests for one_cycle_lr function."""

    def test_start_and_end(self) -> None:
        """Test LR at start and end of training."""
        lr_start = one_cycle_lr(0, 100, lrf=0.01)
        lr_end = one_cycle_lr(100, 100, lrf=0.01)

        assert lr_start == 1.0
        assert abs(lr_end - 0.01) < 0.001

    def test_midpoint(self) -> None:
        """Test LR at midpoint."""
        lr_mid = one_cycle_lr(50, 100, lrf=0.01)
        expected = 0.01 + (1 - 0.01) * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert abs(lr_mid - expected) < 0.001
