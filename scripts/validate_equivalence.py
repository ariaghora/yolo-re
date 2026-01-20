#!/usr/bin/env python3
"""Validate functional equivalence between our model and reference.

Usage:
    python scripts/validate_equivalence.py gelan-c
    python scripts/validate_equivalence.py yolov9-c
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from yolo.model.model import YOLO

# Add reference to path for loading their model (needed for torch.load unpickling)
REPO_ROOT = Path(__file__).parent.parent
REF_PATH = REPO_ROOT / "_reference" / "yolov9"
sys.path.insert(0, str(REF_PATH))


def load_reference_model(ckpt_path: Path) -> torch.nn.Module:
    """Load reference model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    model.eval()
    return model


def validate_gelan_c() -> bool:
    """Validate GELAN-C equivalence."""
    print("=" * 60)
    print("Validating GELAN-C equivalence")
    print("=" * 60)
    
    # Load reference model
    ref_ckpt_path = REPO_ROOT / "_reference" / "weights" / "gelan-c.pt"
    print(f"Loading reference model from: {ref_ckpt_path}")
    ref_model = load_reference_model(ref_ckpt_path)
    
    # Load our model with converted weights
    our_weights_path = REPO_ROOT / "weights" / "gelan-c.pt"
    print(f"Loading our model with weights from: {our_weights_path}")
    our_model = YOLO.from_yaml(REPO_ROOT / "configs" / "models" / "gelan-c.yaml")
    our_model.load_state_dict(torch.load(our_weights_path, map_location="cpu"))
    our_model.eval()
    
    # Test with random input
    print("\nTesting with random input (640x640)...")
    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        ref_out = ref_model(x)
        our_out = our_model(x)
    
    # Reference returns (decoded, raw) in eval mode
    # decoded is [batch, 84, num_anchors] where 84 = 4 (box) + 80 (classes)
    if isinstance(ref_out, tuple):
        ref_decoded = ref_out[0]
    else:
        ref_decoded = ref_out
    
    if isinstance(our_out, tuple):
        our_decoded = our_out[0]
    else:
        our_decoded = our_out
    
    print(f"Reference output shape: {ref_decoded.shape}")
    print(f"Our output shape: {our_decoded.shape}")
    
    # Check equivalence
    atol = 1e-5
    if torch.allclose(our_decoded, ref_decoded, atol=atol):
        print(f"\nPASSED: Outputs match within atol={atol}")
        return True
    else:
        diff = (our_decoded - ref_decoded).abs()
        print("\nFAILED: Outputs do not match")
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")
        return False


def validate_yolov9_c() -> bool:
    """Validate YOLOv9-C equivalence."""
    print("=" * 60)
    print("Validating YOLOv9-C equivalence")
    print("=" * 60)
    
    # Load reference model
    ref_ckpt_path = REPO_ROOT / "_reference" / "weights" / "yolov9-c.pt"
    print(f"Loading reference model from: {ref_ckpt_path}")
    ref_model = load_reference_model(ref_ckpt_path)
    
    # Load our model with converted weights
    our_weights_path = REPO_ROOT / "weights" / "yolov9-c.pt"
    print(f"Loading our model with weights from: {our_weights_path}")
    our_model = YOLO.from_yaml(REPO_ROOT / "configs" / "models" / "yolov9-c.yaml")
    our_model.load_state_dict(torch.load(our_weights_path, map_location="cpu"))
    our_model.eval()
    
    # Test with random input
    print("\nTesting with random input (640x640)...")
    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        ref_out = ref_model(x)
        our_out = our_model(x)
    
    # DualDetect returns different structure
    # Reference: tuple of (list[decoded_aux, decoded_main], list[raw_aux, raw_main])
    # We need to compare the decoded outputs
    if isinstance(ref_out, tuple) and isinstance(ref_out[0], list):
        ref_decoded = ref_out[0]  # [decoded_aux, decoded_main]
    else:
        ref_decoded = ref_out
    
    if isinstance(our_out, tuple) and isinstance(our_out[0], list):
        our_decoded = our_out[0]  # [decoded_aux, decoded_main]
    else:
        our_decoded = our_out
    
    print(f"Reference output type: {type(ref_decoded)}")
    print(f"Our output type: {type(our_decoded)}")
    
    # Check equivalence for each output
    atol = 1e-5
    all_passed = True
    
    if not isinstance(ref_decoded, list) or not isinstance(our_decoded, list):
        raise TypeError("YOLOv9-C should return list outputs")
    
    for i, (ref_d, our_d) in enumerate(zip(ref_decoded, our_decoded)):
        print(f"\nOutput {i}:")
        print(f"  Reference shape: {ref_d.shape}")
        print(f"  Our shape: {our_d.shape}")
        
        if torch.allclose(our_d, ref_d, atol=atol):
            print(f"  PASSED: Match within atol={atol}")
        else:
            diff = (our_d - ref_d).abs()
            print("  FAILED: No match")
            print(f"    Max diff: {diff.max().item():.6e}")
            print(f"    Mean diff: {diff.mean().item():.6e}")
            all_passed = False
    
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model equivalence")
    parser.add_argument(
        "model",
        choices=["gelan-c", "yolov9-c", "all"],
        help="Model to validate",
    )
    args = parser.parse_args()
    
    results = {}
    
    if args.model in ("gelan-c", "all"):
        results["gelan-c"] = validate_gelan_c()
    
    if args.model in ("yolov9-c", "all"):
        results["yolov9-c"] = validate_yolov9_c()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {model}: {status}")
    
    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
