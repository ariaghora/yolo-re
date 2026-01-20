"""Tests for weight loading equivalence with reference implementation."""

import sys
from pathlib import Path

import pytest
import torch

from yolo.blocks import SPPELAN, ADown, Conv, RepNCSPELAN4
from yolo.heads.detect import DetectDFL

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def reference_state_dict():
    """Load the reference state dict (extracted from gelan-c.pt)."""
    path = "_reference/weights/gelan-c-state_dict.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        pytest.skip(f"Reference weights not found at {path}")


@pytest.fixture
def reference_module():
    """Factory fixture to import reference modules."""
    ref_path = Path("_reference/yolov9")
    if not ref_path.exists():
        pytest.skip("Reference yolov9 repo not found")

    def _get_module(name: str):
        sys.path.insert(0, str(ref_path))
        try:
            # Most modules are in common.py
            from models import common  # type: ignore[import-not-found]

            if hasattr(common, name):
                return getattr(common, name)

            # Detection heads are in yolo.py
            from models import yolo  # type: ignore[import-not-found]

            if hasattr(yolo, name):
                return getattr(yolo, name)

            pytest.skip(f"Could not find reference {name}")
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Could not import reference {name}: {e}")
        finally:
            if str(ref_path) in sys.path:
                sys.path.remove(str(ref_path))

    return _get_module


# ============================================================================
# Helper functions for weight key mapping
# ============================================================================


def sync_batchnorm_eps(module: torch.nn.Module, eps: float = 0.001, momentum: float = 0.03) -> None:
    """Pretrained checkpoints have eps=0.001 baked into model objects, but fresh
    reference modules use PyTorch defaults. Without this, outputs differ by ~0.05%.
    """
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = eps
            m.momentum = momentum


def extract_weights(state_dict: dict, prefix: str) -> dict:
    """Extract weights with a given prefix, removing the prefix."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def map_conv_keys(ref_keys: dict) -> dict:
    """Map reference Conv keys to our Conv keys (identical structure)."""
    return ref_keys


def map_repconv_keys(ref_keys: dict) -> dict:
    """Map reference RepConvN keys to our RepConv keys.

    Reference: conv1, conv2 (each is a Conv with conv+bn)
    Ours: conv1, conv2 (same structure)
    """
    return ref_keys


def map_adown_keys(ref_keys: dict) -> dict:
    """Map reference ADown keys to our ADown keys.

    Reference: cv1, cv2
    Ours: conv_stride, conv_pool
    """
    mapped = {}
    for k, v in ref_keys.items():
        if k.startswith("cv1."):
            mapped["conv_stride." + k[4:]] = v
        elif k.startswith("cv2."):
            mapped["conv_pool." + k[4:]] = v
        else:
            mapped[k] = v
    return mapped


def map_sppelan_keys(ref_keys: dict) -> dict:
    """Map reference SPPELAN keys to our SPPELAN keys.

    Reference: cv1, cv2 (SP), cv3 (SP), cv4 (SP), cv5
    Ours: conv_in, pool1, pool2, pool3, conv_out

    Note: SP layers have no weights (just MaxPool2d), so only cv1 and cv5 matter.
    """
    mapped = {}
    for k, v in ref_keys.items():
        if k.startswith("cv1."):
            mapped["conv_in." + k[4:]] = v
        elif k.startswith("cv5."):
            mapped["conv_out." + k[4:]] = v
        else:
            mapped[k] = v
    return mapped


def map_repncsp_keys(ref_keys: dict) -> dict:
    """Map reference RepNCSP keys to our RepNCSP keys.

    Reference: cv1, cv2, cv3, m (Sequential of RepNBottleneck)
    Ours: conv1, conv2, conv3, bottlenecks
    """
    mapped = {}
    for k, v in ref_keys.items():
        if k.startswith("cv1."):
            mapped["conv1." + k[4:]] = v
        elif k.startswith("cv2."):
            mapped["conv2." + k[4:]] = v
        elif k.startswith("cv3."):
            mapped["conv3." + k[4:]] = v
        elif k.startswith("m."):
            # m.0.cv1 -> bottlenecks.0.conv1
            rest = k[2:]  # Remove "m."
            rest = rest.replace(".cv1.", ".conv1.").replace(".cv2.", ".conv2.")
            mapped["bottlenecks." + rest] = v
        else:
            mapped[k] = v
    return mapped


def map_repncspelan4_keys(ref_keys: dict) -> dict:
    """Map reference RepNCSPELAN4 keys to our RepNCSPELAN4 keys.

    Reference: cv1, cv2 (Sequential[RepNCSP, Conv]), cv3 (Sequential[RepNCSP, Conv]), cv4
    Ours: conv_in, block1 (Sequential[RepNCSP, Conv]), block2 (Sequential[RepNCSP, Conv]), conv_out
    """
    mapped = {}
    for k, v in ref_keys.items():
        if k.startswith("cv1."):
            mapped["conv_in." + k[4:]] = v
        elif k.startswith("cv2."):
            # cv2.0 is RepNCSP, cv2.1 is Conv
            rest = k[4:]  # Remove "cv2."
            # Map RepNCSP internal keys
            rest = rest.replace(".cv1.", ".conv1.").replace(".cv2.", ".conv2.").replace(
                ".cv3.", ".conv3."
            ).replace(".m.", ".bottlenecks.")
            mapped["block1." + rest] = v
        elif k.startswith("cv3."):
            rest = k[4:]
            rest = rest.replace(".cv1.", ".conv1.").replace(".cv2.", ".conv2.").replace(
                ".cv3.", ".conv3."
            ).replace(".m.", ".bottlenecks.")
            mapped["block2." + rest] = v
        elif k.startswith("cv4."):
            mapped["conv_out." + k[4:]] = v
        else:
            mapped[k] = v
    return mapped


def map_detectdfl_keys(ref_keys: dict) -> dict:
    """Map reference DDetect keys to our DetectDFL keys.

    Reference: cv2 (ModuleList), cv3 (ModuleList), dfl
    Ours: box_convs, cls_convs, dfl
    """
    mapped = {}
    for k, v in ref_keys.items():
        if k.startswith("cv2."):
            mapped["box_convs." + k[4:]] = v
        elif k.startswith("cv3."):
            mapped["cls_convs." + k[4:]] = v
        else:
            mapped[k] = v
    return mapped


# ============================================================================
# Conv Tests
# ============================================================================


class TestConvEquivalence:
    """Test that our Conv produces identical output to reference given same weights."""

    def test_conv_layer_0(self, reference_state_dict):
        """Test first conv layer (model.0): 3 -> 64 channels, 3x3, stride 2."""
        our_conv = Conv(in_channels=3, out_channels=64, kernel_size=3, stride=2)

        weights = extract_weights(reference_state_dict, "model.0.")
        our_conv.load_state_dict(map_conv_keys(weights))
        our_conv.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            our_output = our_conv(x)

        assert our_output.shape == (1, 64, 320, 320)

    def test_conv_layer_1(self, reference_state_dict):
        """Test second conv layer (model.1): 64 -> 128 channels, 3x3, stride 2."""
        our_conv = Conv(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        weights = extract_weights(reference_state_dict, "model.1.")
        our_conv.load_state_dict(map_conv_keys(weights))
        our_conv.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 64, 320, 320)

        with torch.no_grad():
            our_output = our_conv(x)

        assert our_output.shape == (1, 128, 160, 160)

    def test_conv_output_matches_reference(self, reference_state_dict, reference_module):
        """Test that our Conv output exactly matches reference implementation."""
        RefConv = reference_module("Conv")

        our_conv = Conv(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        ref_conv = RefConv(3, 64, 3, 2)

        weights = extract_weights(reference_state_dict, "model.0.")
        our_conv.load_state_dict(weights)
        ref_conv.load_state_dict(weights)
        sync_batchnorm_eps(ref_conv)  # Match pretrained eps/momentum

        our_conv.eval()
        ref_conv.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            our_output = our_conv(x)
            ref_output = ref_conv(x)

        assert torch.allclose(our_output, ref_output, atol=1e-6), (
            f"Output mismatch! Max diff: {(our_output - ref_output).abs().max()}"
        )


# ============================================================================
# ADown Tests
# ============================================================================


class TestADownEquivalence:
    """Test ADown equivalence with reference."""

    def test_adown_output_matches_reference(self, reference_state_dict, reference_module):
        """Test model.3: ADown 256 -> 256."""
        RefADown = reference_module("ADown")

        # model.3 is ADown: input from model.2 (256ch) -> 256ch
        our_adown = ADown(in_channels=256, out_channels=256)
        ref_adown = RefADown(256, 256)

        ref_weights = extract_weights(reference_state_dict, "model.3.")
        our_weights = map_adown_keys(ref_weights)

        our_adown.load_state_dict(our_weights)
        ref_adown.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_adown)  # Match pretrained eps/momentum

        our_adown.eval()
        ref_adown.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 256, 160, 160)

        with torch.no_grad():
            our_output = our_adown(x)
            ref_output = ref_adown(x)

        assert torch.allclose(our_output, ref_output, atol=1e-6), (
            f"ADown mismatch! Max diff: {(our_output - ref_output).abs().max()}"
        )


# ============================================================================
# SPPELAN Tests
# ============================================================================


class TestSPPELANEquivalence:
    """Test SPPELAN equivalence with reference."""

    def test_sppelan_output_matches_reference(self, reference_state_dict, reference_module):
        """Test model.9: SPPELAN 512 -> 512, hidden=256."""
        RefSPPELAN = reference_module("SPPELAN")

        # model.9 is SPPELAN: 512 -> 512, c3=256
        our_sppelan = SPPELAN(in_channels=512, out_channels=512, hidden_channels=256)
        ref_sppelan = RefSPPELAN(512, 512, 256)

        ref_weights = extract_weights(reference_state_dict, "model.9.")
        our_weights = map_sppelan_keys(ref_weights)

        our_sppelan.load_state_dict(our_weights)
        ref_sppelan.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_sppelan)  # Match pretrained eps/momentum

        our_sppelan.eval()
        ref_sppelan.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 512, 20, 20)

        with torch.no_grad():
            our_output = our_sppelan(x)
            ref_output = ref_sppelan(x)

        assert torch.allclose(our_output, ref_output, atol=1e-6), (
            f"SPPELAN mismatch! Max diff: {(our_output - ref_output).abs().max()}"
        )


# ============================================================================
# RepNCSPELAN4 Tests
# ============================================================================


class TestRepNCSPELAN4Equivalence:
    """Test RepNCSPELAN4 equivalence with reference."""

    def test_repncspelan4_output_matches_reference(self, reference_state_dict, reference_module):
        """Test model.2: RepNCSPELAN4 128->256, hidden=128, block=64."""
        RefRepNCSPELAN4 = reference_module("RepNCSPELAN4")

        # model.2: in=128 (from model.1), out=256, c3=128, c4=64, c5=1
        our_block = RepNCSPELAN4(
            in_channels=128,
            out_channels=256,
            hidden_channels=128,
            block_channels=64,
            num_repeats=1,
        )
        ref_block = RefRepNCSPELAN4(128, 256, 128, 64, 1)

        ref_weights = extract_weights(reference_state_dict, "model.2.")
        our_weights = map_repncspelan4_keys(ref_weights)

        our_block.load_state_dict(our_weights)
        ref_block.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_block)  # Match pretrained eps/momentum

        our_block.eval()
        ref_block.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 128, 160, 160)

        with torch.no_grad():
            our_output = our_block(x)
            ref_output = ref_block(x)

        assert torch.allclose(our_output, ref_output, atol=1e-5), (
            f"RepNCSPELAN4 mismatch! Max diff: {(our_output - ref_output).abs().max()}"
        )

    def test_repncspelan4_stage4(self, reference_state_dict, reference_module):
        """Test model.8: RepNCSPELAN4 512->512, hidden=512, block=256."""
        RefRepNCSPELAN4 = reference_module("RepNCSPELAN4")

        our_block = RepNCSPELAN4(
            in_channels=512,
            out_channels=512,
            hidden_channels=512,
            block_channels=256,
            num_repeats=1,
        )
        ref_block = RefRepNCSPELAN4(512, 512, 512, 256, 1)

        ref_weights = extract_weights(reference_state_dict, "model.8.")
        our_weights = map_repncspelan4_keys(ref_weights)

        our_block.load_state_dict(our_weights)
        ref_block.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_block)  # Match pretrained eps/momentum

        our_block.eval()
        ref_block.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 512, 20, 20)

        with torch.no_grad():
            our_output = our_block(x)
            ref_output = ref_block(x)

        assert torch.allclose(our_output, ref_output, atol=1e-5), (
            f"RepNCSPELAN4 stage4 mismatch! Max diff: {(our_output - ref_output).abs().max()}"
        )


# ============================================================================
# DetectDFL Tests
# ============================================================================


class TestDetectDFLEquivalence:
    """Test DetectDFL equivalence with reference."""

    def test_detectdfl_output_matches_reference(self, reference_state_dict, reference_module):
        """Test model.22: DetectDFL (DDetect in reference)."""
        RefDDetect = reference_module("DDetect")

        # model.22: DDetect with 3 input levels
        # From gelan-c.yaml: inputs are [15, 18, 21] with channels [256, 512, 512]
        in_channels = (256, 512, 512)

        our_detect = DetectDFL(num_classes=80, in_channels=in_channels)
        ref_detect = RefDDetect(nc=80, ch=in_channels)

        ref_weights = extract_weights(reference_state_dict, "model.22.")
        our_weights = map_detectdfl_keys(ref_weights)

        our_detect.load_state_dict(our_weights)
        ref_detect.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_detect)  # Match pretrained eps/momentum

        # Set stride (normally done during model construction)
        stride = torch.tensor([8.0, 16.0, 32.0])
        our_detect.stride = stride
        ref_detect.stride = stride

        our_detect.eval()
        ref_detect.eval()

        torch.manual_seed(42)
        # Create inputs matching the expected sizes at each stride level
        x = [
            torch.randn(1, 256, 80, 80),  # P3: 640/8 = 80
            torch.randn(1, 512, 40, 40),  # P4: 640/16 = 40
            torch.randn(1, 512, 20, 20),  # P5: 640/32 = 20
        ]

        with torch.no_grad():
            our_output = our_detect(x.copy())
            ref_output = ref_detect(x.copy())

        # Compare decoded outputs (first element of tuple in eval mode)
        our_decoded = our_output[0]
        ref_decoded = ref_output[0]

        assert torch.allclose(our_decoded, ref_decoded, atol=1e-5), (
            f"DetectDFL mismatch! Max diff: {(our_decoded - ref_decoded).abs().max()}"
        )

    def test_detectdfl_training_output_matches_reference(
        self, reference_state_dict, reference_module
    ):
        """Test DetectDFL training mode output matches reference."""
        RefDDetect = reference_module("DDetect")

        in_channels = (256, 512, 512)

        our_detect = DetectDFL(num_classes=80, in_channels=in_channels)
        ref_detect = RefDDetect(nc=80, ch=in_channels)

        ref_weights = extract_weights(reference_state_dict, "model.22.")
        our_weights = map_detectdfl_keys(ref_weights)

        our_detect.load_state_dict(our_weights)
        ref_detect.load_state_dict(ref_weights)
        sync_batchnorm_eps(ref_detect)  # Match pretrained eps/momentum

        our_detect.train()
        ref_detect.train()

        torch.manual_seed(42)
        x = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]

        with torch.no_grad():
            our_output = our_detect(x.copy())
            ref_output = ref_detect(x.copy())

        # In training mode, output is list of tensors
        for i, (our_level, ref_level) in enumerate(zip(our_output, ref_output)):
            assert torch.allclose(our_level, ref_level, atol=1e-5), (
                f"DetectDFL training level {i} mismatch! "
                f"Max diff: {(our_level - ref_level).abs().max()}"
            )
