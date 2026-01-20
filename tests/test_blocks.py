"""Tests for YOLO building blocks."""

import torch

from yolo.blocks import (
    SPPELAN,
    ADown,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    RepConv,
    RepNBottleneck,
    RepNCSP,
    RepNCSPELAN4,
    Silence,
)


class TestConv:
    def test_forward_shape(self):
        conv = Conv(64, 128, kernel_size=3, stride=1)
        x = torch.randn(1, 64, 32, 32)
        y = conv(x)
        assert y.shape == (1, 128, 32, 32)

    def test_forward_stride(self):
        conv = Conv(64, 128, kernel_size=3, stride=2)
        x = torch.randn(1, 64, 32, 32)
        y = conv(x)
        assert y.shape == (1, 128, 16, 16)

    def test_from_config(self):
        cfg = Conv.Config(in_channels=32, out_channels=64, kernel_size=3)
        conv = Conv.from_config(cfg)
        x = torch.randn(1, 32, 16, 16)
        y = conv(x)
        assert y.shape == (1, 64, 16, 16)


class TestRepConv:
    def test_forward_shape(self):
        conv = RepConv(64, 64)
        x = torch.randn(1, 64, 32, 32)
        y = conv(x)
        assert y.shape == (1, 64, 32, 32)


class TestRepNBottleneck:
    def test_forward_shape(self):
        block = RepNBottleneck(64, 64)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        assert y.shape == (1, 64, 32, 32)

    def test_shortcut(self):
        # With shortcut (default)
        block = RepNBottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        assert y.shape == (1, 64, 32, 32)

    def test_no_shortcut_different_channels(self):
        block = RepNBottleneck(64, 128, shortcut=True)  # shortcut disabled due to channel mismatch
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        assert y.shape == (1, 128, 32, 32)


class TestRepNCSP:
    def test_forward_shape(self):
        block = RepNCSP(64, 64, num_repeats=2)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        assert y.shape == (1, 64, 32, 32)


class TestConcat:
    def test_forward(self):
        concat = Concat(dimension=1)
        x1 = torch.randn(1, 32, 16, 16)
        x2 = torch.randn(1, 64, 16, 16)
        y = concat([x1, x2])
        assert y.shape == (1, 96, 16, 16)


class TestSilence:
    def test_forward(self):
        silence = Silence()
        x = torch.randn(1, 64, 32, 32)
        y = silence(x)
        assert torch.equal(x, y)


class TestADown:
    def test_forward_shape(self):
        # ADown halves spatial dimensions
        block = ADown(128, 256)
        x = torch.randn(1, 128, 32, 32)
        y = block(x)
        assert y.shape == (1, 256, 16, 16)

    def test_from_config(self):
        cfg = ADown.Config(in_channels=64, out_channels=128)
        block = ADown.from_config(cfg)
        x = torch.randn(1, 64, 64, 64)
        y = block(x)
        assert y.shape == (1, 128, 32, 32)


class TestSPPELAN:
    def test_forward_shape(self):
        # SPPELAN preserves spatial dimensions
        block = SPPELAN(512, 512, 256)
        x = torch.randn(1, 512, 20, 20)
        y = block(x)
        assert y.shape == (1, 512, 20, 20)

    def test_from_config(self):
        cfg = SPPELAN.Config(in_channels=256, out_channels=256, hidden_channels=128)
        block = SPPELAN.from_config(cfg)
        x = torch.randn(1, 256, 16, 16)
        y = block(x)
        assert y.shape == (1, 256, 16, 16)


class TestRepNCSPELAN4:
    def test_forward_shape(self):
        # From gelan-c.yaml: RepNCSPELAN4, [256, 128, 64, 1]
        # Args: c1=in, c2=out, c3=hidden, c4=block_ch, c5=num_repeats
        block = RepNCSPELAN4(128, 256, 128, 64, 1)
        x = torch.randn(1, 128, 32, 32)
        y = block(x)
        assert y.shape == (1, 256, 32, 32)

    def test_forward_shape_larger(self):
        # From gelan-c.yaml: RepNCSPELAN4, [512, 256, 128, 1]
        block = RepNCSPELAN4(256, 512, 256, 128, 1)
        x = torch.randn(1, 256, 16, 16)
        y = block(x)
        assert y.shape == (1, 512, 16, 16)

    def test_from_config(self):
        cfg = RepNCSPELAN4.Config(
            in_channels=64, out_channels=128, hidden_channels=64, block_channels=32
        )
        block = RepNCSPELAN4.from_config(cfg)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        assert y.shape == (1, 128, 32, 32)


class TestCBLinear:
    def test_forward_shape(self):
        # CBLinear splits output into multiple tensors
        block = CBLinear(512, [256, 512, 512])
        x = torch.randn(1, 512, 20, 20)
        outputs = block(x)
        assert len(outputs) == 3
        assert outputs[0].shape == (1, 256, 20, 20)
        assert outputs[1].shape == (1, 512, 20, 20)
        assert outputs[2].shape == (1, 512, 20, 20)

    def test_forward_single_output(self):
        block = CBLinear(256, [256])
        x = torch.randn(1, 256, 32, 32)
        outputs = block(x)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 256, 32, 32)


class TestCBFuse:
    def test_forward_shape(self):
        # CBFuse takes a single list: [cb_out1, cb_out2, ..., target]
        # idx specifies which tensor from each CBLinear tuple to use
        fuse = CBFuse([0, 0])

        # Simulate CBLinear outputs (tuples) - all selected tensors must have same channels
        cb1_out = (torch.randn(1, 256, 40, 40), torch.randn(1, 512, 40, 40))
        cb2_out = (torch.randn(1, 256, 20, 20), torch.randn(1, 512, 20, 20))
        target = torch.randn(1, 256, 10, 10)

        # idx=[0, 0] means select first tensor (256ch) from each tuple
        # Input is a single list with cb_outputs followed by target
        y = fuse([cb1_out, cb2_out, target])
        # Output should match target size
        assert y.shape == (1, 256, 10, 10)

    def test_forward_single_input(self):
        fuse = CBFuse([2])
        cb_out = (
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 512, 40, 40),
        )
        target = torch.randn(1, 512, 20, 20)
        # Single cb_output followed by target
        y = fuse([cb_out, target])
        assert y.shape == (1, 512, 20, 20)
