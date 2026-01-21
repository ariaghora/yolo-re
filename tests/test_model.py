"""Tests for model construction."""

from pathlib import Path

import torch

from yolo.model import YOLO, ModelConfig, parse_yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "models"


class TestParseYAML:
    """Tests for YAML parsing."""

    def test_parse_gelan_c(self):
        """Test parsing gelan-c.yaml."""
        config = parse_yaml(CONFIGS_DIR / "gelan-c.yaml")

        assert config.num_classes == 80
        assert config.depth_multiplier == 1.0
        assert config.width_multiplier == 1.0
        assert len(config.layers) > 0

    def test_layer_names_unique(self):
        """Test that all layer names are unique."""
        config = parse_yaml(CONFIGS_DIR / "gelan-c.yaml")
        names = [layer["name"] for layer in config.layers]
        assert len(names) == len(set(names)), "Layer names must be unique"


class TestYOLOConstruction:
    """Tests for YOLO model construction."""

    def test_from_yaml(self):
        """Test building model from YAML."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        assert isinstance(model, YOLO)
        assert len(model.layers) > 0

    def test_forward_shape_training(self):
        """Test forward pass shape in training mode."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        model.train()

        x = torch.randn(1, 3, 640, 640)
        out = model(x)

        # Training returns list of predictions per level
        assert isinstance(out, list)
        assert len(out) == 3  # P3, P4, P5

        # Each level: [B, num_outputs, H, W]
        # num_outputs = num_classes + reg_max * 4 = 80 + 16*4 = 144
        assert out[0].shape[0] == 1
        assert out[0].shape[1] == 144

    def test_forward_shape_inference(self):
        """Test forward pass shape in eval mode."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        # Inference returns (decoded, raw)
        assert isinstance(out, tuple)
        decoded, raw = out

        # Decoded: [B, num_outputs, num_anchors]
        assert decoded.shape[0] == 1
        # 4 (box) + 80 (class) = 84
        assert decoded.shape[1] == 84

    def test_different_input_sizes(self):
        """Test model works with different input sizes."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        model.eval()

        for size in [320, 416, 640]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out[0].shape[0] == 1

    def test_batch_size(self):
        """Test model works with different batch sizes."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        model.eval()

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 640, 640)
            with torch.no_grad():
                out = model(x)
            assert out[0].shape[0] == batch_size


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = ModelConfig()
        assert config.num_classes == 80
        assert config.depth_multiplier == 1.0
        assert config.width_multiplier == 1.0
        assert config.layers == []

    def test_custom_values(self):
        """Test custom config values."""
        config = ModelConfig(
            num_classes=10,
            depth_multiplier=0.5,
            width_multiplier=0.25,
        )
        assert config.num_classes == 10
        assert config.depth_multiplier == 0.5
        assert config.width_multiplier == 0.25


class TestOptimGroups:
    """Tests for YOLO.optim_groups()."""

    def test_returns_three_groups(self):
        """Test that optim_groups returns 3 param groups."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        groups = model.optim_groups()

        assert len(groups) == 3

    def test_weight_decay_assignment(self):
        """Test weight decay is only on conv/linear weights."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        groups = model.optim_groups(weight_decay=0.01)

        # Group 0: weights (has decay)
        # Group 1: bn (no decay)
        # Group 2: bias (no decay)
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0
        assert groups[2]["weight_decay"] == 0.0

    def test_params_are_included(self):
        """Test param groups contain parameters."""
        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        groups = model.optim_groups()

        total_in_groups = 0
        for g in groups:
            params = g["params"]
            if isinstance(params, list):
                total_in_groups += len(params)

        # Should have a reasonable number of params
        assert total_in_groups > 100

    def test_works_with_optimizer(self):
        """Test groups work with actual optimizer."""
        from torch.optim import AdamW

        model = YOLO.from_yaml(CONFIGS_DIR / "gelan-c.yaml")
        groups = model.optim_groups(weight_decay=0.0005)

        optimizer = AdamW(groups, lr=0.01)
        assert len(optimizer.param_groups) == 3
