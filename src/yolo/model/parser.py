"""YAML model config parser."""

from copy import deepcopy
from pathlib import Path

import torch.nn as nn
import yaml

from yolo.blocks.auxiliary import CBFuse, CBLinear
from yolo.blocks.common import Concat, Silence
from yolo.blocks.conv import Conv
from yolo.blocks.downsample import ADown
from yolo.blocks.gelan import RepNCSPELAN4
from yolo.blocks.sppelan import SPPELAN
from yolo.heads.detect import DetectDFL, DualDetectDFL
from yolo.model.config import LayerDef, ModelConfig


def parse_yaml(path: str | Path) -> ModelConfig:
    """Parse model config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    model_data = data.get("model", {})
    return ModelConfig(
        num_classes=model_data.get("num_classes", 80),
        depth_multiplier=model_data.get("depth_multiplier", 1.0),
        width_multiplier=model_data.get("width_multiplier", 1.0),
        layers=data.get("layers", []),
    )


def _apply_width_multiplier(value: int, multiplier: float, divisor: int = 8) -> int:
    """Apply width multiplier and round to nearest divisor.

    Args:
        value: Original channel count.
        multiplier: Width scaling factor.
        divisor: Round to nearest multiple of this value.

    Returns:
        Scaled channel count, rounded to divisor.
    """
    if multiplier == 1.0:
        return value
    scaled = value * multiplier
    return max(divisor, int(scaled + divisor / 2) // divisor * divisor)


def _apply_depth_multiplier(value: int, multiplier: float) -> int:
    """Apply depth multiplier to repeat count.

    Args:
        value: Original repeat count.
        multiplier: Depth scaling factor.

    Returns:
        Scaled repeat count, minimum 1.
    """
    if multiplier == 1.0:
        return value
    return max(1, round(value * multiplier))


class ModelBuilder:
    """Builds model layers from config.

    Tracks channel dimensions and connections as layers are added.
    """

    def __init__(
        self,
        num_classes: int,
        width_mult: float,
        depth_mult: float,
        input_channels: int = 3,
    ):
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.depth_mult = depth_mult

        self.layers: dict[str, nn.Module] = {}
        self.connections: dict[str, str | list[str]] = {}
        self.channel_map: dict[str, int] = {"input": input_channels}
        self.prev_name = "input"
        self.detect_inputs: list[str] = []

    def add_layer(self, layer_def: LayerDef) -> None:
        """Add a layer from definition."""
        name = layer_def.name
        block_type = layer_def.type
        from_layers = layer_def.from_layers if layer_def.from_layers else self.prev_name
        params = deepcopy(layer_def.params)

        self.connections[name] = from_layers

        if isinstance(from_layers, list):
            in_channels_list = [self.channel_map[n] for n in from_layers]
        elif isinstance(from_layers, str):
            in_channels_list = [self.channel_map[from_layers]]
        else:
            raise TypeError(f"from_layers must be str or list[str], got {type(from_layers)}")

        if block_type in ("DetectDFL", "DualDetectDFL"):
            block, out_ch = self._build_detect(block_type, in_channels_list)
            self.detect_inputs = from_layers if isinstance(from_layers, list) else [from_layers]
        elif block_type == "Concat":
            block, out_ch = self._build_concat(params, in_channels_list)
        elif block_type == "Silence":
            block, out_ch = Silence(), in_channels_list[0]
        elif block_type == "Upsample":
            block, out_ch = self._build_upsample(params, in_channels_list[0])
        elif block_type == "CBLinear":
            block, out_ch = self._build_cblinear(params, in_channels_list[0])
        elif block_type == "CBFuse":
            block, out_ch = self._build_cbfuse(params, in_channels_list)
        else:
            block, out_ch = self._build_standard(block_type, params, in_channels_list[0])

        self.layers[name] = block
        self.channel_map[name] = out_ch
        self.prev_name = name

    def _build_detect(
        self, block_type: str, in_channels_list: list[int]
    ) -> tuple[nn.Module, int]:
        """Build detection head.

        Args:
            block_type: Either "DetectDFL" or "DualDetectDFL".
            in_channels_list: Input channels from each feature level.

        Returns:
            Detection module and output channels (0 for detect heads).
        """
        in_ch = tuple(in_channels_list)
        if block_type == "DetectDFL":
            return DetectDFL(self.num_classes, in_ch), 0
        elif block_type == "DualDetectDFL":
            return DualDetectDFL(self.num_classes, in_ch), 0
        else:
            raise ValueError(f"Unknown detect type: {block_type}")

    def _build_concat(
        self, params: dict, in_channels_list: list[int]
    ) -> tuple[nn.Module, int]:
        """Build concat layer.

        Args:
            params: Layer parameters (may include 'dimension').
            in_channels_list: Channel counts from each input tensor.

        Returns:
            Concat module and sum of input channels.
        """
        dimension = params.get("dimension", 1)
        return Concat(dimension), sum(in_channels_list)

    def _build_upsample(self, params: dict, in_channels: int) -> tuple[nn.Module, int]:
        """Build upsample layer.

        Args:
            params: Layer parameters (scale_factor, mode).
            in_channels: Input channel count (unchanged by upsample).

        Returns:
            Upsample module and unchanged channel count.
        """
        scale_factor = params.get("scale_factor", 2)
        mode = params.get("mode", "nearest")
        return nn.Upsample(scale_factor=scale_factor, mode=mode), in_channels

    def _build_cblinear(self, params: dict, in_channels: int) -> tuple[nn.Module, int]:
        """Build CBLinear layer for auxiliary supervision.

        Args:
            params: Must include 'out_channels_list'.
            in_channels: Input channel count.

        Returns:
            CBLinear module and last output channel count.
        """
        out_channels_list = params["out_channels_list"]
        out_channels_list = [
            _apply_width_multiplier(c, self.width_mult) for c in out_channels_list
        ]
        return CBLinear(in_channels, out_channels_list), out_channels_list[-1]

    def _build_cbfuse(
        self, params: dict, in_channels_list: list[int]
    ) -> tuple[nn.Module, int]:
        """Build CBFuse layer for auxiliary supervision.

        Args:
            params: Must include 'idx' for fusion indices.
            in_channels_list: Channel counts from inputs.

        Returns:
            CBFuse module and last input's channel count.
        """
        idx = params["idx"]
        return CBFuse(idx), in_channels_list[-1]

    def _build_standard(
        self, block_type: str, params: dict, in_channels: int
    ) -> tuple[nn.Module, int]:
        """Build standard block (Conv, ADown, RepNCSPELAN4, SPPELAN).

        Args:
            block_type: Block class name.
            params: Block-specific parameters.
            in_channels: Input channel count.

        Returns:
            Block module and output channel count.
        """
        for p in ["out_channels", "hidden_channels", "block_channels"]:
            if p in params:
                params[p] = _apply_width_multiplier(params[p], self.width_mult)

        if "num_repeats" in params:
            params["num_repeats"] = _apply_depth_multiplier(
                params["num_repeats"], self.depth_mult
            )

        params["in_channels"] = in_channels

        if block_type == "Conv":
            block = Conv(**params)
        elif block_type == "ADown":
            block = ADown(**params)
        elif block_type == "RepNCSPELAN4":
            block = RepNCSPELAN4(**params)
        elif block_type == "SPPELAN":
            block = SPPELAN(**params)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        return block, params["out_channels"]

    def build(self) -> tuple[nn.ModuleDict, dict[str, str | list[str]], list[str]]:
        """Return built layers, connections, and detect inputs.

        Returns:
            Tuple of (layers ModuleDict, connection map, detect input names).
        """
        return nn.ModuleDict(self.layers), self.connections, self.detect_inputs


def build_layers(
    config: ModelConfig,
    input_channels: int = 3,
) -> tuple[nn.ModuleDict, dict[str, str | list[str]], list[str]]:
    """Build model layers from parsed config.

    This is the main entry point for constructing a model from YAML config.
    Handles channel inference, width/depth multipliers, and layer connections.

    Args:
        config: Parsed ModelConfig from YAML.
        input_channels: Number of input image channels (default 3 for RGB).

    Returns:
        Tuple of:
            - layers: OrderedDict of named nn.Module layers
            - connections: Map of layer name to input layer name(s)
            - detect_inputs: Names of layers feeding into detection head
    """
    builder = ModelBuilder(
        num_classes=config.num_classes,
        width_mult=config.width_multiplier,
        depth_mult=config.depth_multiplier,
        input_channels=input_channels,
    )

    for layer_dict in config.layers:
        layer_def = LayerDef.from_dict(deepcopy(layer_dict))
        builder.add_layer(layer_def)

    return builder.build()
