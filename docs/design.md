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
└── train/            # Training hyperparameters
    └── default.yaml
```

Each config type has a corresponding dataclass in `src/yolo/`. YAML files are optional. Users can construct configs programmatically.

## Usage Patterns

### From YAML configs

```python
from yolo import YOLO, DataConfig, TrainConfig, Trainer

model = YOLO.from_yaml("configs/models/gelan-c.yaml")
data = DataConfig.from_yaml("configs/data/coco.yaml")
config = TrainConfig.from_yaml("configs/train/default.yaml")

trainer = Trainer(model, data, config)
trainer.train()
```

### Mix YAML and code

```python
from yolo import YOLO, DataConfig, Trainer

model = YOLO.from_yaml("configs/models/gelan-c.yaml")
data = DataConfig(train_path="/my/data", num_classes=10)
trainer = Trainer(model, data, epochs=100)
trainer.train()
```

### Fully programmatic

```python
from yolo import YOLO, DataConfig, TrainConfig, Trainer

model = YOLO.from_yaml("configs/models/gelan-c.yaml")
data = DataConfig(train_path="/my/data", num_classes=10)
config = TrainConfig(epochs=100, lr=0.01)
trainer = Trainer(model, data, config)
trainer.train()
```

## Component Boundaries

| Component | Responsibility | Replaceable? |
|-----------|---------------|--------------|
| Model | Forward pass, architecture | Yes - any nn.Module |
| Loss | Compute loss from preds + targets | Yes - pass `loss_fn` to Trainer |
| Optimizer | Parameter updates | Yes - pass `optimizer` to Trainer |
| Data | Load images + annotations | Yes - any DataLoader |
| Trainer | Training loop, checkpoints, logging | Yes - use your own |

Loss is decoupled from model. This is a departure from the reference where they're tangled.

## Defaults

Every config field has a sensible default except truly required ones:
- `data.train_path` - required (no sane default)
- `model.num_classes` - required (must match dataset)

Everything else works out of the box.

## Optimizer

Weight decay should not apply to biases and batchnorm. YOLO provides `model.optim_groups()` to handle this correctly:

```python
from torch.optim import AdamW

optimizer = AdamW(model.optim_groups(weight_decay=0.0005), lr=0.01)
trainer = Trainer(model, data, optimizer=optimizer)
```

If you don't pass an optimizer, Trainer uses SGD with correct param groups:

```python
trainer = Trainer(model, data)  # Uses SGD internally
```

If you want full control with your own param grouping:

```python
optimizer = MyOptimizer(my_custom_groups, lr=0.001)
trainer = Trainer(model, data, optimizer=optimizer)
```
