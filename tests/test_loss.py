"""Tests for loss functions."""

import torch

from yolo.loss import IoUType, LossConfig, TALoss, bbox_iou, make_anchors


class TestBboxIoU:
    """Tests for IoU calculations."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1."""
        box = torch.tensor([[10, 10, 20, 20]])
        iou = bbox_iou(box, box, xywh=False)
        assert iou.item() == 1.0

    def test_non_overlapping_boxes(self):
        """Non-overlapping boxes should have IoU of 0."""
        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[20, 20, 30, 30]])
        iou = bbox_iou(box1, box2, xywh=False)
        assert iou.item() == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes."""
        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[5, 5, 15, 15]])
        iou = bbox_iou(box1, box2, xywh=False)
        expected = 25 / (100 + 100 - 25)
        assert abs(iou.item() - expected) < 1e-5

    def test_ciou(self):
        """CIoU should be less than or equal to IoU."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        iou = bbox_iou(box1, box2, xywh=False)
        ciou = bbox_iou(box1, box2, xywh=False, iou_type=IoUType.CIOU)
        assert ciou.item() <= iou.item()

    def test_iou_types(self):
        """Test all IoU types compute without error."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        for iou_type in IoUType:
            result = bbox_iou(box1, box2, xywh=False, iou_type=iou_type)
            assert result.shape == (1, 1)

    def test_xywh_format(self):
        """Test xywh input format."""
        box1_xywh = torch.tensor([[15.0, 15.0, 10.0, 10.0]])
        box2_xywh = torch.tensor([[15.0, 15.0, 10.0, 10.0]])
        iou = bbox_iou(box1_xywh, box2_xywh, xywh=True)
        assert abs(iou.item() - 1.0) < 1e-5


class TestMakeAnchors:
    """Tests for anchor generation."""

    def test_single_scale(self):
        """Test anchor generation for single feature map."""
        feat = torch.randn(1, 256, 80, 80)
        anchor_points, stride_tensor = make_anchors([feat], [8], 0.5)

        assert anchor_points.shape == (80 * 80, 2)
        assert stride_tensor.shape == (80 * 80, 1)
        assert (stride_tensor == 8).all()

        assert anchor_points[0, 0].item() == 0.5
        assert anchor_points[0, 1].item() == 0.5

    def test_multi_scale(self):
        """Test anchor generation for multiple feature maps."""
        feats = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        strides = [8, 16, 32]
        anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)

        total = 80 * 80 + 40 * 40 + 20 * 20
        assert anchor_points.shape == (total, 2)
        assert stride_tensor.shape == (total, 1)


class TestTALoss:
    """Tests for TALoss."""

    def test_smoke_single_head(self):
        """Loss computes without error for single head."""
        num_classes = 80
        reg_max = 16
        strides = [8, 16, 32]

        loss_fn = TALoss(num_classes, reg_max, strides)
        no = reg_max * 4 + num_classes

        feats = [
            torch.randn(2, no, 80, 80),
            torch.randn(2, no, 40, 40),
            torch.randn(2, no, 20, 20),
        ]

        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
            [0, 1, 0.3, 0.3, 0.2, 0.2],
            [1, 0, 0.7, 0.7, 0.15, 0.15],
        ])

        total_loss, loss_items = loss_fn(feats, targets)

        assert total_loss.shape == ()
        assert loss_items.shape == (3,)
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

    def test_smoke_dual_head(self):
        """Loss computes without error for dual head."""
        num_classes = 80
        reg_max = 16
        strides = [8, 16, 32]

        loss_fn = TALoss(num_classes, reg_max, strides)
        no = reg_max * 4 + num_classes

        feats_aux = [
            torch.randn(2, no, 80, 80),
            torch.randn(2, no, 40, 40),
            torch.randn(2, no, 20, 20),
        ]
        feats_main = [
            torch.randn(2, no, 80, 80),
            torch.randn(2, no, 40, 40),
            torch.randn(2, no, 20, 20),
        ]

        decoded = torch.randn(2, 80 * 80 + 40 * 40 + 20 * 20, num_classes + 4)
        preds = (decoded, (feats_aux, feats_main))

        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.1, 0.1],
            [1, 0, 0.7, 0.7, 0.15, 0.15],
        ])

        total_loss, loss_items = loss_fn(preds, targets)

        assert total_loss.shape == ()
        assert loss_items.shape == (3,)
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

    def test_empty_targets(self):
        """Loss handles empty targets gracefully."""
        num_classes = 80
        reg_max = 16
        strides = [8, 16, 32]

        loss_fn = TALoss(num_classes, reg_max, strides)
        no = reg_max * 4 + num_classes

        feats = [
            torch.randn(2, no, 80, 80),
            torch.randn(2, no, 40, 40),
            torch.randn(2, no, 20, 20),
        ]

        targets = torch.zeros(0, 6)

        total_loss, loss_items = loss_fn(feats, targets)

        assert total_loss.shape == ()
        assert not torch.isnan(total_loss)

    def test_custom_config(self):
        """Loss respects custom config."""
        config = LossConfig(box_gain=10.0, cls_gain=1.0, dfl_gain=2.0)
        loss_fn = TALoss(80, 16, [8, 16, 32], config)

        assert loss_fn.config.box_gain == 10.0
        assert loss_fn.config.cls_gain == 1.0
        assert loss_fn.config.dfl_gain == 2.0

    def test_label_smoothing(self):
        """Label smoothing affects loss computation."""
        num_classes = 80
        reg_max = 16
        strides = [8, 16, 32]
        no = reg_max * 4 + num_classes

        feats = [
            torch.randn(1, no, 20, 20),
            torch.randn(1, no, 10, 10),
            torch.randn(1, no, 5, 5),
        ]
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])

        # Without label smoothing
        loss_fn_no_smooth = TALoss(num_classes, reg_max, strides, LossConfig(label_smoothing=0.0))
        loss_no_smooth, _ = loss_fn_no_smooth(feats, targets)

        # With label smoothing
        loss_fn_smooth = TALoss(num_classes, reg_max, strides, LossConfig(label_smoothing=0.1))
        loss_smooth, _ = loss_fn_smooth(feats, targets)

        # Losses should differ when label smoothing is applied
        assert loss_no_smooth.item() != loss_smooth.item()

    def test_gradient_flow(self):
        """Gradients flow through loss computation."""
        num_classes = 80
        reg_max = 16
        strides = [8, 16, 32]

        loss_fn = TALoss(num_classes, reg_max, strides)
        no = reg_max * 4 + num_classes

        feats = [
            torch.randn(1, no, 20, 20, requires_grad=True),
            torch.randn(1, no, 10, 10, requires_grad=True),
            torch.randn(1, no, 5, 5, requires_grad=True),
        ]

        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])

        total_loss, _ = loss_fn(feats, targets)
        total_loss.backward()

        for feat in feats:
            assert feat.grad is not None
            assert not torch.isnan(feat.grad).any()
