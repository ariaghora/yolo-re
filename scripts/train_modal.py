#!/usr/bin/env python3
"""Train YOLO on modal.com with GPU.

Usage:
    modal run scripts/train_modal.py
    modal run scripts/train_modal.py --epochs 50 --batch-size 32
    modal run scripts/train_modal.py --eval-pretrained  # Test pretrained weights
    modal run scripts/train_modal.py --use-pretrained --augment-preset light  # Fine-tune

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
    .apt_install("wget", "unzip", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        # Core
        "torch",
        "torchvision",
        "pyyaml",
        "numpy",
        "pillow",
        "albumentations",
        "tqdm",
        "opencv-python",
        # All yolov9 dependencies for weight unpickling
        "gitpython",
        "ipython",
        "matplotlib",
        "psutil",
        "requests",
        "scipy",
        "thop",
        "tensorboard",
        "pandas",
        "seaborn",
        "pycocotools",
    )
    .run_commands(
        # Clone yolov9 reference repo for weight unpickling
        "git clone --depth 1 https://github.com/WongKinYiu/yolov9.git /root/yolov9"
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


def download_and_convert_weights(weights_dir: str = "/vol/weights") -> str:
    """Download and convert pretrained gelan-c weights if not present."""
    import sys
    import urllib.request
    from pathlib import Path

    import torch

    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    converted_file = weights_path / "gelan-c-converted.pt"
    if converted_file.exists():
        print(f"Converted weights already exist at {converted_file}")
        return str(converted_file)

    # Download original weights
    raw_file = weights_path / "gelan-c-raw.pt"
    if not raw_file.exists():
        url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt"
        print(f"Downloading pretrained weights from {url}...")
        urllib.request.urlretrieve(url, raw_file)

    # Load and convert weights
    print("Converting weights to our format...")
    
    # Need to add reference path for unpickling
    sys.path.insert(0, "/root/yolov9")
    
    ckpt = torch.load(raw_file, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ref_model = ckpt["model"].float()
        ref_sd = ref_model.state_dict()
    else:
        ref_sd = ckpt

    # Convert state dict keys from reference format to our format
    # Reference uses model.0, model.1, etc. We use named layers.
    from yolo import YOLO
    our_model = YOLO.from_yaml("/root/configs/models/gelan-c.yaml")
    our_sd = our_model.state_dict()
    
    # Simple key mapping by matching shapes
    new_sd = {}
    ref_keys = list(ref_sd.keys())
    our_keys = list(our_sd.keys())
    
    # Map by position (both models have same structure)
    for ref_key, our_key in zip(ref_keys, our_keys):
        if ref_sd[ref_key].shape == our_sd[our_key].shape:
            new_sd[our_key] = ref_sd[ref_key]
        else:
            ref_shape, our_shape = ref_sd[ref_key].shape, our_sd[our_key].shape
            print(f"Shape mismatch: {ref_key} {ref_shape} vs {our_key} {our_shape}")
    
    # Verify all keys mapped
    if len(new_sd) == len(our_sd):
        print(f"Successfully mapped {len(new_sd)} parameters")
        torch.save(new_sd, converted_file)
        return str(converted_file)
    else:
        raise RuntimeError(f"Weight conversion failed: {len(new_sd)}/{len(our_sd)} keys mapped")


def _eval_pretrained_impl() -> dict:
    """Evaluate pretrained weights on COCO128 to verify inference pipeline.

    Returns:
        Evaluation metrics.
    """
    import torch

    from yolo import YOLO
    from yolo.data.config import DataConfig
    from yolo.data.dataset import create_dataloader
    from yolo.eval.evaluator import Evaluator

    # Download dataset and weights
    train_path = download_coco128(f"{VOLUME_PATH}/data")
    weights_file = download_and_convert_weights(f"{VOLUME_PATH}/weights")

    # Load model with pretrained weights
    print("Loading pretrained gelan-c weights...")
    model = YOLO.from_yaml("/root/configs/models/gelan-c.yaml")
    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create validation dataloader
    data = DataConfig(
        train_path=train_path,
        val_path=train_path,
        num_classes=80,
        batch_size=16,
        workers=4,
    )
    val_loader = create_dataloader(data, train=False)

    # Run evaluation
    print("Running evaluation...")
    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        num_classes=80,
        device="cuda",
        debug_dir=f"{VOLUME_PATH}/runs/pretrained_debug",
    )

    metrics = evaluator.evaluate(epoch=0)

    # Commit volume to persist debug images
    volume.commit()

    print("\n=== PRETRAINED EVAL RESULTS ===")
    print(f"mAP@50: {metrics['map50']:.4f}")
    print(f"mAP@75: {metrics['map75']:.4f}")
    print(f"mAP@50:95: {metrics['map']:.4f}")
    print(f"Debug images saved to: {VOLUME_PATH}/runs/pretrained_debug/")

    return metrics


def _train_impl(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.01,
    val_period: int = 5,
    use_pretrained: bool = False,
    augment_preset: str = "full",
) -> dict:
    """Train YOLO on COCO128.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_period: Validate every N epochs.
        use_pretrained: If True, start from pretrained weights.
        augment_preset: Augmentation preset ("full", "light", "minimal").

    Returns:
        Training metrics.
    """
    import torch

    from yolo import YOLO, Trainer
    from yolo.data.config import AugmentConfig, DataConfig

    # Download dataset (cached in volume)
    train_path = download_coco128(f"{VOLUME_PATH}/data")

    # Load model
    model = YOLO.from_yaml("/root/configs/models/gelan-c.yaml")
    
    if use_pretrained:
        weights_file = download_and_convert_weights(f"{VOLUME_PATH}/weights")
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights")

    # Create augmentation config from preset
    augment = AugmentConfig.from_preset(augment_preset)  # type: ignore[arg-type]
    print(f"Using augmentation preset: {augment_preset}")
    print(f"  mosaic={augment.mosaic}, mixup={augment.mixup}")

    # COCO128 uses same images for train/val (it's a smoke test dataset)
    data = DataConfig(
        train_path=train_path,
        val_path=train_path,
        num_classes=80,
        batch_size=batch_size,
        workers=4,
        augment=augment,
    )

    trainer = Trainer(
        model,
        data,
        epochs=epochs,
        lr=lr,
        val_period=val_period,
        output_dir=f"{VOLUME_PATH}/runs",
        device="cuda",
    )

    trainer.train()

    # Commit volume to persist checkpoints
    volume.commit()

    return {
        "epochs": epochs,
        "best_map50": trainer.best_fitness,
        "checkpoint": f"{VOLUME_PATH}/runs/last.pt",
    }


# GPU-specific function wrappers
@app.function(gpu="T4", volumes={VOLUME_PATH: volume}, timeout=7200)
def eval_pretrained_t4() -> dict:
    return _eval_pretrained_impl()


@app.function(gpu="T4", volumes={VOLUME_PATH: volume}, timeout=7200)
def train_t4(
    epochs: int,
    batch_size: int,
    lr: float,
    val_period: int,
    use_pretrained: bool = False,
    augment_preset: str = "full",
) -> dict:
    return _train_impl(epochs, batch_size, lr, val_period, use_pretrained, augment_preset)


@app.function(gpu="L4", volumes={VOLUME_PATH: volume}, timeout=7200)
def train_l4(
    epochs: int,
    batch_size: int,
    lr: float,
    val_period: int,
    use_pretrained: bool = False,
    augment_preset: str = "full",
) -> dict:
    return _train_impl(epochs, batch_size, lr, val_period, use_pretrained, augment_preset)


@app.function(gpu="A10G", volumes={VOLUME_PATH: volume}, timeout=7200)
def train_a10g(
    epochs: int,
    batch_size: int,
    lr: float,
    val_period: int,
    use_pretrained: bool = False,
    augment_preset: str = "full",
) -> dict:
    return _train_impl(epochs, batch_size, lr, val_period, use_pretrained, augment_preset)


@app.function(gpu="A100", volumes={VOLUME_PATH: volume}, timeout=7200)
def train_a100(
    epochs: int,
    batch_size: int,
    lr: float,
    val_period: int,
    use_pretrained: bool = False,
    augment_preset: str = "full",
) -> dict:
    return _train_impl(epochs, batch_size, lr, val_period, use_pretrained, augment_preset)


@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.01,
    val_period: int = 5,
    gpu: str = "T4",
    eval_pretrained: bool = False,
    use_pretrained: bool = False,
    augment_preset: str = "full",
):
    """Local entrypoint for modal run.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_period: Validate every N epochs.
        gpu: GPU type (T4, L4, A10G, A100).
        eval_pretrained: If True, evaluate pretrained weights instead of training.
        use_pretrained: If True, start training from pretrained weights.
        augment_preset: Augmentation preset ("full", "light", "minimal").
            Use "full" for training from scratch.
            Use "light" or "minimal" when fine-tuning pretrained weights.
    """
    if eval_pretrained:
        print("Evaluating pretrained weights on COCO128...")
        result = eval_pretrained_t4.remote()
        print(f"Evaluation complete: {result}")
        return

    # Auto-adjust LR for fine-tuning (0.01 is too high for pretrained weights)
    if use_pretrained and lr == 0.01:
        lr = 0.001
        print(f"Auto-adjusted LR to {lr} for fine-tuning (override with --lr)")

    print(f"Starting training: {epochs} epochs, batch {batch_size}, lr {lr}")
    print(f"Validation every {val_period} epochs, GPU: {gpu}")
    print(f"Augmentation preset: {augment_preset}")
    if use_pretrained:
        print("Starting from pretrained weights")

    train_fn = {
        "T4": train_t4,
        "L4": train_l4,
        "A10G": train_a10g,
        "A100": train_a100,
    }.get(gpu.upper())

    if train_fn is None:
        raise ValueError(f"Unknown GPU: {gpu}. Choose from T4, L4, A10G, A100")

    result = train_fn.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_period=val_period,
        use_pretrained=use_pretrained,
        augment_preset=augment_preset,
    )
    print(f"Training complete: {result}")
