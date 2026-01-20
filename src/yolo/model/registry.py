"""Block type registry for YAML parsing."""

import torch.nn as nn

from yolo.blocks.auxiliary import CBFuse, CBLinear
from yolo.blocks.common import Concat, Silence
from yolo.blocks.conv import Conv
from yolo.blocks.downsample import ADown
from yolo.blocks.gelan import RepNCSPELAN4
from yolo.blocks.sppelan import SPPELAN
from yolo.heads.detect import DetectDFL, DualDetectDFL

# Map type names to block classes
BLOCKS: dict[str, type[nn.Module]] = {
    "Conv": Conv,
    "ADown": ADown,
    "RepNCSPELAN4": RepNCSPELAN4,
    "SPPELAN": SPPELAN,
    "Concat": Concat,
    "Silence": Silence,
    "CBLinear": CBLinear,
    "CBFuse": CBFuse,
    "DetectDFL": DetectDFL,
    "DualDetectDFL": DualDetectDFL,
    "Upsample": nn.Upsample,
}


def get_block_class(name: str) -> type[nn.Module]:
    """Get block class by name.

    Args:
        name: Block type name.

    Returns:
        Block class.

    Raises:
        KeyError: If block type not found.
    """
    if name not in BLOCKS:
        raise KeyError(f"Unknown block type: {name}. Available: {list(BLOCKS.keys())}")
    return BLOCKS[name]
