# YOLO, Reimplemented

A YOLO implementation you can actually read.

Research code shouldn't be production code. This project takes working YOLO implementations and rewrites them with type annotations, typed configs, clear boundaries, and readable structure. Users can replace any component, but shouldn't have to. Out of the box, this thing trains.

Does this have equal feature parity? Maybe no.
Will this reproduce the reference result fully? I don't know. Maybe? Training from pretrained checkpoint on COCO128 dataset reduces losses and increases mAPs. Somehow the model learns.

> **Note:** Currently based on [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9). Future versions may incorporate other YOLO variants.

## Installation

```bash
uv pip install git+https://github.com/ariaghora/yolo-re.git
```

## Usage

```python
from yolo import YOLO, Trainer
from yolo.data.config import DataConfig

model = YOLO.from_yaml("configs/models/gelan-c.yaml")
data = DataConfig(train_path="/path/to/images", num_classes=80)

trainer = Trainer(model, data, epochs=100)
trainer.train()
```

Load pretrained weights:

```python
model = YOLO.from_yaml("configs/models/gelan-c.yaml")
model.load_state_dict(torch.load("weights.pt"))
```

See [docs/design.md](docs/design.md) for config patterns and component boundaries.

## Development

```bash
git clone https://github.com/ariaghora/yolo-re.git
cd yolo-re
uv sync
uv run pytest
```
