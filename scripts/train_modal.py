#!/usr/bin/env python3
"""Train YOLO on modal.com with GPU.

Usage:
    modal run scripts/train_modal.py
    modal run scripts/train_modal.py --epochs 50 --batch-size 32

Requirements:
    uv pip install -e ".[modal]"
    modal token new  # first time setup
"""

import modal

# Volume for storing checkpoints and data
volume = modal.Volume.from_name("yolo-training", create_if_missing=True)
VOLUME_PATH = "/vol"

# Image with dependencies + local source
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "pyyaml",
        "numpy",
        "pillow",
        "albumentations",
        "tqdm",
        "opencv-python",
    )
    .add_local_python_source("yolo")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("yolo-training", image=image)


def download_coco128(data_dir: str = "/vol/data") -> str:
    """Download COCO128 dataset if not present."""
    import shutil
    import urllib.request
    import zipfile
    from pathlib import Path

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    dataset_path = data_path / "coco128"
    if dataset_path.exists() and (dataset_path / "images" / "train").exists():
        print(f"COCO128 already exists at {dataset_path}")
        return str(dataset_path / "images" / "train")

    url = "https://ultralytics.com/assets/coco128.zip"
    zip_path = data_path / "coco128.zip"

    print(f"Downloading COCO128 from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting to {data_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    zip_path.unlink()

    # Reorganize directory structure
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    train_images = images_dir / "train"
    train_labels = labels_dir / "train"

    if not train_images.exists() and (images_dir / "train2017").exists():
        shutil.move(str(images_dir / "train2017"), str(train_images))

    if not train_labels.exists() and (labels_dir / "train2017").exists():
        shutil.move(str(labels_dir / "train2017"), str(train_labels))

    print(f"COCO128 ready: {len(list(train_images.glob('*.jpg')))} images")
    return str(train_images)


@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def train(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.01,
) -> dict:
    """Train YOLO on COCO128.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.

    Returns:
        Training metrics.
    """
    from yolo import YOLO, Trainer
    from yolo.data.config import DataConfig

    # Download dataset (cached in volume)
    train_path = download_coco128(f"{VOLUME_PATH}/data")

    # Load model
    model = YOLO.from_yaml("/root/configs/models/gelan-c.yaml")

    data = DataConfig(
        train_path=train_path,
        num_classes=80,
        batch_size=batch_size,
        workers=4,
    )

    trainer = Trainer(
        model,
        data,
        epochs=epochs,
        lr=lr,
        output_dir=f"{VOLUME_PATH}/runs",
        device="cuda",
    )

    trainer.train()

    # Commit volume to persist checkpoints
    volume.commit()

    return {
        "epochs": epochs,
        "checkpoint": f"{VOLUME_PATH}/runs/last.pt",
    }


@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.01,
):
    """Local entrypoint for modal run.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
    """
    print(f"Starting training: {epochs} epochs, batch {batch_size}, lr {lr}")
    result = train.remote(epochs=epochs, batch_size=batch_size, lr=lr)
    print(f"Training complete: {result}")
