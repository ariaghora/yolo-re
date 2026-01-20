"""Tests for detection heads."""

import torch

from yolo.heads import DetectDFL, DualDetectDFL
from yolo.heads.anchor import dist2bbox, make_anchors
from yolo.heads.dfl import DFL


class TestDFL:
    def test_forward_shape(self):
        dfl = DFL(16)
        # Input: [batch, 4*reg_max, anchors]
        x = torch.randn(2, 64, 100)
        y = dfl(x)
        # Output: [batch, 4, anchors]
        assert y.shape == (2, 4, 100)

    def test_forward_values(self):
        # DFL computes weighted average of [0, 1, ..., 15]
        dfl = DFL(16)
        # If softmax output is uniform, result should be ~7.5 (mean of 0-15)
        # But with random input, just check it runs and has reasonable range
        x = torch.randn(1, 64, 10)
        y = dfl(x)
        assert y.min() >= 0  # Output should be non-negative (weighted avg of 0-15)
        assert y.max() <= 15


class TestMakeAnchors:
    def test_output_shape(self):
        # 3 feature levels with different sizes
        features = [
            torch.randn(1, 256, 80, 80),  # P3
            torch.randn(1, 512, 40, 40),  # P4
            torch.randn(1, 512, 20, 20),  # P5
        ]
        strides = torch.tensor([8, 16, 32])

        anchors, stride_tensor = make_anchors(features, strides)

        # Total anchors = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        assert anchors.shape == (8400, 2)
        assert stride_tensor.shape == (8400, 1)

    def test_anchor_values(self):
        features = [torch.randn(1, 256, 4, 4)]
        strides = torch.tensor([8])

        anchors, stride_tensor = make_anchors(features, strides, grid_cell_offset=0.5)

        # With offset 0.5, first anchor should be at (0.5, 0.5)
        assert torch.allclose(anchors[0], torch.tensor([0.5, 0.5]))
        # All strides should be 8
        assert (stride_tensor == 8).all()


class TestDist2Bbox:
    def test_xywh_output(self):
        # distance: [batch, 4, anchors] as ltrb
        distance = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
        anchor_points = torch.tensor([[5.0, 5.0], [10.0, 10.0]])

        boxes = dist2bbox(distance, anchor_points, xywh=True, dim=1)

        # For anchor (5, 5) with ltrb (1, 1, 1, 1):
        # x1y1 = (4, 4), x2y2 = (6, 6)
        # center = (5, 5), wh = (2, 2)
        assert boxes.shape == (1, 4, 2)

    def test_xyxy_output(self):
        distance = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])
        anchor_points = torch.tensor([[5.0, 5.0]])

        boxes = dist2bbox(distance, anchor_points, xywh=False, dim=1)

        # x1y1 = (4, 4), x2y2 = (6, 6)
        expected = torch.tensor([[[4.0], [4.0], [6.0], [6.0]]])
        assert torch.allclose(boxes, expected)


class TestDetectDFL:
    def test_forward_training(self):
        head = DetectDFL(num_classes=80, in_channels=(256, 512, 512))
        head.train()

        # 3 feature levels
        features = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        out = head(features)

        # Training returns list of raw predictions
        assert isinstance(out, list)
        assert len(out) == 3
        # Each output: [batch, num_outputs, H, W]
        # num_outputs = 80 + 16*4 = 144
        assert out[0].shape == (2, 144, 80, 80)
        assert out[1].shape == (2, 144, 40, 40)
        assert out[2].shape == (2, 144, 20, 20)

    def test_forward_inference(self):
        head = DetectDFL(num_classes=80, in_channels=(256, 512, 512))
        head.eval()
        head.stride = torch.tensor([8, 16, 32])

        features = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        with torch.no_grad():
            y, x = head(features)

        # Inference returns (decoded, raw)
        # decoded: [batch, 4+num_classes, total_anchors]
        total_anchors = 80 * 80 + 40 * 40 + 20 * 20
        assert y.shape == (2, 84, total_anchors)
        # raw is list of 3 tensors
        assert len(x) == 3


class TestDualDetectDFL:
    def test_forward_training(self):
        # 6 channels: 3 aux + 3 main
        head = DualDetectDFL(num_classes=80, in_channels=(256, 512, 512, 256, 512, 512))
        head.train()

        # 6 feature maps
        features = [
            torch.randn(2, 256, 80, 80),  # aux P3
            torch.randn(2, 512, 40, 40),  # aux P4
            torch.randn(2, 512, 20, 20),  # aux P5
            torch.randn(2, 256, 80, 80),  # main P3
            torch.randn(2, 512, 40, 40),  # main P4
            torch.randn(2, 512, 20, 20),  # main P5
        ]

        out = head(features)

        # Training returns [aux_preds, main_preds]
        assert isinstance(out, list)
        assert len(out) == 2
        assert len(out[0]) == 3  # aux has 3 levels
        assert len(out[1]) == 3  # main has 3 levels

    def test_forward_inference(self):
        head = DualDetectDFL(num_classes=80, in_channels=(256, 512, 512, 256, 512, 512))
        head.eval()
        head.stride = torch.tensor([8, 16, 32])

        features = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 512, 20, 20),
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

        with torch.no_grad():
            y, x = head(features)

        # y is [decoded_aux, decoded_main]
        total_anchors = 80 * 80 + 40 * 40 + 20 * 20
        assert len(y) == 2
        assert y[0].shape == (2, 84, total_anchors)
        assert y[1].shape == (2, 84, total_anchors)
