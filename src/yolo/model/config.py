"""Model configuration dataclasses."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Full model configuration.

    Attributes:
        num_classes: Number of detection classes.
        depth_multiplier: Multiplier for block depth (num_repeats).
        width_multiplier: Multiplier for channel widths.
        layers: List of layer definitions (raw dicts from YAML).
    """

    num_classes: int = 80
    depth_multiplier: float = 1.0
    width_multiplier: float = 1.0
    layers: list[dict] = field(default_factory=list)


@dataclass
class LayerDef:
    """Single layer definition.

    Attributes:
        name: Unique layer identifier.
        type: Block type name (must exist in registry).
        from_layers: Input layer name(s). None means previous layer.
        params: Block-specific parameters (passed to block's Config).
    """

    name: str
    type: str
    from_layers: str | list[str] | None = None
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "LayerDef":
        """Create LayerDef from YAML dict."""
        name = data.pop("name")
        layer_type = data.pop("type")
        from_layers = data.pop("from", None)
        return cls(name=name, type=layer_type, from_layers=from_layers, params=data)
