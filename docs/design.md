# Design

Key design decisions for this YOLO reimplementation.

## Philosophy

Library, not framework. Users can replace any component, but shouldn't have to. Out of the box, this thing trains.

## Config Structure

```
configs/
├── models/           # Model architecture definitions
│   ├── gelan-c.yaml
│   └── yolov9-c.yaml
├── data/             # Dataset configs (paths, class names)
│   └── coco.yaml
└── train/            # Full training configs (combines all)
    └── default.yaml
```

Each config type has a corresponding dataclass in `src/yolo/`. YAML files are optional. Users can construct configs programmatically.

## Usage Patterns

### Minimal (just works)

```python
from yolo import Trainer

trainer = Trainer.from_yaml("configs/train/default.yaml")
trainer.train()
```

### Custom (mix YAML and code)

```python
from yolo import YOLO, DataConfig, Trainer

model = YOLO.from_yaml("configs/models/gelan-c.yaml")
data = DataConfig(train_path="/my/data", num_classes=10)
trainer = Trainer(model=model, data=data, epochs=100)
trainer.train()
```

### Fully programmatic

```python
from yolo import YOLO, DataConfig, TrainConfig, Trainer

model = YOLO(...)  # Build model directly
data = DataConfig(...)
config = TrainConfig(epochs=100, lr=0.01, ...)
trainer = Trainer(model=model, data=data, config=config)
trainer.train()
```

## Component Boundaries

| Component | Responsibility | Replaceable? |
|-----------|---------------|--------------|
| Model | Forward pass, architecture | Yes - any nn.Module |
| Loss | Compute loss from preds + targets | Yes - any callable |
| Data | Load images + annotations | Yes - any DataLoader |
| Trainer | Training loop, checkpoints, logging | Yes - use your own |

Loss is decoupled from model. This is a departure from the reference where they're tangled.

## Config Composition

Train configs can reference other configs by path:

```yaml
# configs/train/default.yaml
model: configs/models/gelan-c.yaml
data: configs/data/coco.yaml
epochs: 300
lr: 0.01
```

Or embed them inline:

```yaml
# configs/train/custom.yaml
model:
  num_classes: 10
  depth_multiplier: 0.5
data:
  train_path: /path/to/train
  val_path: /path/to/val
epochs: 100
```

## Defaults

Every config field has a sensible default except truly required ones:
- `data.train_path` - required (no sane default)
- `model.num_classes` - required (must match dataset)

Everything else works out of the box.
