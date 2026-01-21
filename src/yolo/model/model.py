"""YOLO model class."""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from yolo.heads.detect import DetectDFL, DualDetectDFL
from yolo.model.config import ModelConfig
from yolo.model.parser import build_layers, parse_yaml

# Type alias for layer outputs:
# - Tensor: most layers
# - tuple[Tensor, ...]: CBLinear
# - tuple[Tensor, list[Tensor]]: detect heads (training mode)
_LayerOutput = Tensor | tuple[Tensor, ...] | tuple[Tensor, list[Tensor]]


class YOLO(nn.Module):
    """YOLO detection model.

    Supports both programmatic and YAML-based construction.

    Example:
        # From YAML
        model = YOLO.from_yaml("configs/models/gelan-c.yaml")

        # Programmatic
        model = YOLO(layers, connections, detect_inputs)
    """

    def __init__(
        self,
        layers: nn.ModuleDict,
        connections: dict[str, str | list[str]],
        detect_inputs: list[str],
    ):
        """Initialize YOLO model.

        Args:
            layers: Ordered dict of named layers.
            connections: Map of layer name -> input layer name(s).
            detect_inputs: Names of layers that feed into detection head.
        """
        super().__init__()
        self.layers = layers
        self.connections = connections
        self.detect_inputs = detect_inputs
        self._save_names = self._compute_save_names()
        self._stride_initialized = False

    def _compute_save_names(self) -> set[str]:
        """Compute which layer outputs need to be saved for skip connections."""
        save = set()
        for from_spec in self.connections.values():
            if isinstance(from_spec, str):
                save.add(from_spec)
            elif isinstance(from_spec, list):
                save.update(from_spec)
            else:
                raise TypeError(f"connection must be str or list[str], got {type(from_spec)}")
        return save

    def _get_layer_input(
        self,
        name: str,
        outputs: dict[str, _LayerOutput],
    ) -> _LayerOutput | list[_LayerOutput]:
        """Get input tensor(s) for a layer.

        Args:
            name: Current layer name.
            outputs: Map of layer name to output tensor.

        Returns:
            Single tensor or list of tensors/tuples for layers with multiple inputs.
        """
        from_spec = self.connections[name]
        if isinstance(from_spec, str):
            return outputs[from_spec]
        elif isinstance(from_spec, list):
            return [outputs[n] for n in from_spec]
        else:
            raise TypeError(f"connection must be str or list[str], got {type(from_spec)}")

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            During training: Raw predictions from detection head.
            During inference: Decoded predictions.
        """
        outputs: dict[str, _LayerOutput] = {"input": x}
        out: Tensor | tuple[Tensor, list[Tensor]] = x

        for name, layer in self.layers.items():
            inp = self._get_layer_input(name, outputs)
            out = layer(inp)

            if name in self._save_names or name == list(self.layers.keys())[-1]:
                outputs[name] = out

        return out

    def init_stride(self, input_size: int = 256) -> None:
        """Initialize detection head stride by running a forward pass.

        Args:
            input_size: Size of dummy input for stride computation.
        """
        if self._stride_initialized:
            return

        detect: DetectDFL | DualDetectDFL | None = None
        detect_name: str | None = None
        for name, layer in self.layers.items():
            if isinstance(layer, DetectDFL | DualDetectDFL):
                detect = layer
                detect_name = name
                break

        if detect is None:
            return

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            outputs: dict[str, _LayerOutput] = {"input": dummy}

            for name, layer in self.layers.items():
                if name == detect_name:
                    from_spec = self.connections[name]
                    if not isinstance(from_spec, list):
                        raise ValueError("Detect head must have multiple inputs")
                    raw_inputs = [outputs[n] for n in from_spec]

                    feats: list[Tensor] = []
                    for inp in raw_inputs:
                        if not isinstance(inp, Tensor):
                            raise TypeError(f"Detect input must be Tensor, got {type(inp)}")
                        feats.append(inp)

                    # DualDetectDFL inputs are [aux..., main...], use main half for strides
                    if isinstance(detect, DualDetectDFL):
                        feats = feats[detect.num_levels :]

                    strides = torch.tensor([
                        input_size / f.shape[-1] for f in feats
                    ])
                    detect.stride = strides
                    break

                inp = self._get_layer_input(name, outputs)
                out = layer(inp)
                outputs[name] = out

        detect.init_bias()
        self._stride_initialized = True
        self.train()

    def optim_groups(
        self, weight_decay: float = 0.0005
    ) -> list[dict[str, list[nn.Parameter] | float]]:
        """Get parameter groups with proper weight decay settings.

        Weight decay should not apply to biases or normalization layers.
        This method returns groups suitable for optimizer constructors.

        Args:
            weight_decay: Weight decay for conv/linear weights.

        Returns:
            List of param group dicts:
            - Group 0: Conv/Linear weights (with weight_decay)
            - Group 1: BatchNorm weights (no decay)
            - Group 2: All biases (no decay)

        Example:
            optimizer = AdamW(model.optim_groups(0.0005), lr=0.01)
        """
        g_weights: list[nn.Parameter] = []
        g_bn: list[nn.Parameter] = []
        g_bias: list[nn.Parameter] = []

        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                g_bias.append(module.bias)

            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                    g_bn.append(module.weight)
            elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                g_weights.append(module.weight)

        return [
            {"params": g_weights, "weight_decay": weight_decay},
            {"params": g_bn, "weight_decay": 0.0},
            {"params": g_bias, "weight_decay": 0.0},
        ]

    @classmethod
    def from_config(cls, config: ModelConfig, input_channels: int = 3) -> "YOLO":
        """Build model from config.

        Args:
            config: Model configuration.
            input_channels: Number of input image channels.

        Returns:
            YOLO model.
        """
        layers, connections, detect_inputs = build_layers(config, input_channels)
        model = cls(layers, connections, detect_inputs)
        model.init_stride()
        return model

    @classmethod
    def from_yaml(cls, path: str | Path, input_channels: int = 3) -> "YOLO":
        """Build model from YAML config file.

        Args:
            path: Path to YAML config.
            input_channels: Number of input image channels.

        Returns:
            YOLO model.
        """
        config = parse_yaml(path)
        return cls.from_config(config, input_channels)
