#!/usr/bin/env python3
"""Convert YOLOv9 reference weights to our format.

Usage:
    python scripts/convert_weights.py gelan-c
    python scripts/convert_weights.py yolov9-c
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

# Add reference to path for unpickling their model objects
REPO_ROOT = Path(__file__).parent.parent
REF_PATH = REPO_ROOT / "_reference" / "yolov9"
sys.path.insert(0, str(REF_PATH))


# Layer index to our layer name mapping for GELAN-C
GELAN_C_LAYERS = {
    0: ("stem1", "Conv"),
    1: ("stem2", "Conv"),
    2: ("stage1", "RepNCSPELAN4"),
    3: ("down1", "ADown"),
    4: ("stage2", "RepNCSPELAN4"),
    5: ("down2", "ADown"),
    6: ("stage3", "RepNCSPELAN4"),
    7: ("down3", "ADown"),
    8: ("stage4", "RepNCSPELAN4"),
    9: ("spp", "SPPELAN"),
    # 10: up1 - no weights
    # 11: concat1 - no weights
    12: ("fpn1", "RepNCSPELAN4"),
    # 13: up2 - no weights
    # 14: concat2 - no weights
    15: ("fpn2", "RepNCSPELAN4"),
    16: ("pan_down1", "ADown"),
    # 17: concat3 - no weights
    18: ("pan1", "RepNCSPELAN4"),
    19: ("pan_down2", "ADown"),
    # 20: concat4 - no weights
    21: ("pan2", "RepNCSPELAN4"),
    22: ("detect", "DetectDFL"),
}

# Layer index to our layer name mapping for YOLOv9-C
# Note: index 0 is Silence (no weights), so keys start from 1
YOLOV9_C_LAYERS = {
    # Main backbone
    1: ("stem1", "Conv"),
    2: ("stem2", "Conv"),
    3: ("stage1", "RepNCSPELAN4"),
    4: ("down1", "ADown"),
    5: ("stage2", "RepNCSPELAN4"),
    6: ("down2", "ADown"),
    7: ("stage3", "RepNCSPELAN4"),
    8: ("down3", "ADown"),
    9: ("stage4", "RepNCSPELAN4"),
    # Neck
    10: ("spp", "SPPELAN"),
    # 11: up1 - no weights
    # 12: concat1 - no weights
    13: ("fpn1", "RepNCSPELAN4"),
    # 14: up2 - no weights
    # 15: concat2 - no weights
    16: ("fpn2", "RepNCSPELAN4"),
    17: ("pan_down1", "ADown"),
    # 18: concat3 - no weights
    19: ("pan1", "RepNCSPELAN4"),
    20: ("pan_down2", "ADown"),
    # 21: concat4 - no weights
    22: ("pan2", "RepNCSPELAN4"),
    # Auxiliary branch routing
    23: ("cb_route1", "CBLinear"),
    24: ("cb_route2", "CBLinear"),
    25: ("cb_route3", "CBLinear"),
    # Auxiliary backbone
    26: ("aux_stem1", "Conv"),
    27: ("aux_stem2", "Conv"),
    28: ("aux_stage1", "RepNCSPELAN4"),
    29: ("aux_down1", "ADown"),
    # 30: aux_fuse1 - CBFuse, no weights
    31: ("aux_stage2", "RepNCSPELAN4"),
    32: ("aux_down2", "ADown"),
    # 33: aux_fuse2 - CBFuse, no weights
    34: ("aux_stage3", "RepNCSPELAN4"),
    35: ("aux_down3", "ADown"),
    # 36: aux_fuse3 - CBFuse, no weights
    37: ("aux_stage4", "RepNCSPELAN4"),
    # Detection head
    38: ("detect", "DualDetectDFL"),
}


def map_conv_key(key: str, layer_name: str) -> str:
    """Map Conv keys (identity - structure matches)."""
    # model.0.conv.weight -> layers.stem1.conv.weight
    parts = key.split(".", 2)  # ["model", "0", "conv.weight"]
    return f"layers.{layer_name}.{parts[2]}"


def map_adown_key(key: str, layer_name: str) -> str:
    """Map ADown keys: cv1->conv_stride, cv2->conv_pool."""
    parts = key.split(".", 2)
    rest = parts[2]
    rest = rest.replace("cv1.", "conv_stride.")
    rest = rest.replace("cv2.", "conv_pool.")
    return f"layers.{layer_name}.{rest}"


def map_sppelan_key(key: str, layer_name: str) -> str:
    """Map SPPELAN keys: cv1->conv_in, cv5->conv_out."""
    parts = key.split(".", 2)
    rest = parts[2]
    rest = rest.replace("cv1.", "conv_in.")
    rest = rest.replace("cv5.", "conv_out.")
    return f"layers.{layer_name}.{rest}"


def map_repncspelan4_key(key: str, layer_name: str) -> str:
    """Map RepNCSPELAN4 keys with nested RepNCSP structure.
    
    Reference structure:
        cv1 -> conv_in (top-level)
        cv2.0.cv1 -> block1.0.conv1 (RepNCSP.conv1)
        cv2.0.cv2 -> block1.0.conv2 (RepNCSP.conv2)
        cv2.0.cv3 -> block1.0.conv3 (RepNCSP.conv3)
        cv2.0.m.N.cv1 -> block1.0.bottlenecks.N.conv1 (RepNBottleneck)
        cv2.0.m.N.cv2 -> block1.0.bottlenecks.N.conv2 (RepNBottleneck)
        cv2.1 -> block1.1 (Conv after RepNCSP, structure identical)
        cv3.* -> block2.* (same as cv2)
        cv4 -> conv_out (top-level)
    """
    parts = key.split(".", 2)
    rest = parts[2]
    
    # Top level RepNCSPELAN4: cv1->conv_in, cv4->conv_out
    if rest.startswith("cv1."):
        rest = "conv_in." + rest[4:]
    elif rest.startswith("cv4."):
        rest = "conv_out." + rest[4:]
    elif rest.startswith("cv2."):
        rest = "block1." + rest[4:]
    elif rest.startswith("cv3."):
        rest = "block2." + rest[4:]
    
    # Now handle RepNCSP internals (inside block1.0. or block2.0.)
    # block1.0.cv1 -> block1.0.conv1
    # block1.0.cv2 -> block1.0.conv2
    # block1.0.cv3 -> block1.0.conv3
    # block1.0.m.N.cv1 -> block1.0.bottlenecks.N.conv1
    # block1.0.m.N.cv2 -> block1.0.bottlenecks.N.conv2
    for block_prefix in ["block1.0.", "block2.0."]:
        if rest.startswith(block_prefix):
            suffix = rest[len(block_prefix):]
            # Handle bottlenecks (m.N.cv1 -> bottlenecks.N.conv1)
            if suffix.startswith("m."):
                suffix = "bottlenecks." + suffix[2:]
                suffix = suffix.replace(".cv1.", ".conv1.")
                suffix = suffix.replace(".cv2.", ".conv2.")
            else:
                # RepNCSP top-level convs
                if suffix.startswith("cv1."):
                    suffix = "conv1." + suffix[4:]
                elif suffix.startswith("cv2."):
                    suffix = "conv2." + suffix[4:]
                elif suffix.startswith("cv3."):
                    suffix = "conv3." + suffix[4:]
            rest = block_prefix + suffix
            break
    
    return f"layers.{layer_name}.{rest}"


def map_detectdfl_key(key: str, layer_name: str) -> str:
    """Map DetectDFL keys: cv2->box_convs, cv3->cls_convs."""
    parts = key.split(".", 2)
    rest = parts[2]
    rest = rest.replace("cv2.", "box_convs.")
    rest = rest.replace("cv3.", "cls_convs.")
    return f"layers.{layer_name}.{rest}"


def map_dualdetectdfl_key(key: str, layer_name: str) -> str:
    """Map DualDetectDFL keys: cv2->aux_box, cv3->aux_cls, cv4->main_box, cv5->main_cls."""
    parts = key.split(".", 2)
    rest = parts[2]
    rest = rest.replace("cv2.", "aux_box_convs.")
    rest = rest.replace("cv3.", "aux_cls_convs.")
    rest = rest.replace("cv4.", "main_box_convs.")
    rest = rest.replace("cv5.", "main_cls_convs.")
    return f"layers.{layer_name}.{rest}"


def map_cblinear_key(key: str, layer_name: str) -> str:
    """Map CBLinear keys (identity - structure matches)."""
    parts = key.split(".", 2)
    return f"layers.{layer_name}.{parts[2]}"


def convert_state_dict(
    ref_sd: dict[str, torch.Tensor],
    layer_mapping: dict[int, tuple[str, str]],
) -> OrderedDict[str, torch.Tensor]:
    """Convert reference state dict to our format."""
    new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    
    for ref_key, tensor in ref_sd.items():
        if not ref_key.startswith("model."):
            continue
        
        # Extract layer index
        parts = ref_key.split(".")
        try:
            layer_idx = int(parts[1])
        except ValueError:
            print(f"Skipping key with non-integer index: {ref_key}")
            continue
        
        # Skip layers without weights (Upsample, Concat, Silence, CBFuse)
        if layer_idx not in layer_mapping:
            continue
        
        layer_name, layer_type = layer_mapping[layer_idx]
        
        # Apply appropriate key transformation
        if layer_type == "Conv":
            new_key = map_conv_key(ref_key, layer_name)
        elif layer_type == "ADown":
            new_key = map_adown_key(ref_key, layer_name)
        elif layer_type == "SPPELAN":
            new_key = map_sppelan_key(ref_key, layer_name)
        elif layer_type == "RepNCSPELAN4":
            new_key = map_repncspelan4_key(ref_key, layer_name)
        elif layer_type == "DetectDFL":
            new_key = map_detectdfl_key(ref_key, layer_name)
        elif layer_type == "DualDetectDFL":
            new_key = map_dualdetectdfl_key(ref_key, layer_name)
        elif layer_type == "CBLinear":
            new_key = map_cblinear_key(ref_key, layer_name)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        new_sd[new_key] = tensor
    
    return new_sd


def load_reference_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load reference checkpoint and extract state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    if "model" in ckpt:
        model = ckpt["model"]
        if hasattr(model, "state_dict"):
            return model.float().state_dict()
        elif isinstance(model, dict):
            return model
    
    # Might be a raw state dict
    if isinstance(ckpt, dict) and any(k.startswith("model.") for k in ckpt.keys()):
        return ckpt
    
    raise ValueError(f"Cannot extract state dict from checkpoint: {path}")


def convert_gelan_c(ref_path: Path, out_path: Path) -> None:
    """Convert GELAN-C weights."""
    print(f"Loading reference checkpoint: {ref_path}")
    ref_sd = load_reference_checkpoint(ref_path)
    print(f"Reference state dict keys: {len(ref_sd)}")
    
    print("Converting keys...")
    new_sd = convert_state_dict(ref_sd, GELAN_C_LAYERS)
    print(f"Converted state dict keys: {len(new_sd)}")
    
    print(f"Saving to: {out_path}")
    torch.save(new_sd, out_path)
    print("Done!")


def convert_yolov9_c(ref_path: Path, out_path: Path) -> None:
    """Convert YOLOv9-C weights."""
    print(f"Loading reference checkpoint: {ref_path}")
    ref_sd = load_reference_checkpoint(ref_path)
    print(f"Reference state dict keys: {len(ref_sd)}")
    
    print("Converting keys...")
    new_sd = convert_state_dict(ref_sd, YOLOV9_C_LAYERS)
    print(f"Converted state dict keys: {len(new_sd)}")
    
    print(f"Saving to: {out_path}")
    torch.save(new_sd, out_path)
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLOv9 weights")
    parser.add_argument(
        "model",
        choices=["gelan-c", "yolov9-c"],
        help="Model to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "weights",
        help="Output directory for converted weights",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == "gelan-c":
        ref_path = REPO_ROOT / "_reference" / "weights" / "gelan-c.pt"
        out_path = args.output_dir / "gelan-c.pt"
        convert_gelan_c(ref_path, out_path)
    elif args.model == "yolov9-c":
        ref_path = REPO_ROOT / "_reference" / "weights" / "yolov9-c.pt"
        out_path = args.output_dir / "yolov9-c.pt"
        convert_yolov9_c(ref_path, out_path)


if __name__ == "__main__":
    main()
