#!/usr/bin/env python3
"""Train YOLO on modal.com with GPU.

Usage:
    modal run scripts/train_modal.py
    modal run scripts/train_modal.py --epochs 50 --batch-size 32
    GPU=A100 modal run scripts/train_modal.py --dataset coco --epochs 300
    modal run scripts/train_modal.py --use-pretrained --augment-preset light

Requirements:
    uv pip install -e ".[modal]"
    modal token new
"""

import os

import modal

volume = modal.Volume.from_name("yolo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch",
        "torchvision",
        "pyyaml",
        "numpy",
        "pillow",
        "albumentations",
        "tqdm",
        "opencv-python",
        # yolov9 dependencies needed for weight unpickling
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
    .run_commands("git clone --depth 1 https://github.com/WongKinYiu/yolov9.git /root/yolov9")
    .add_local_python_source("yolo")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("yolo-training", image=image)


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with progress bar."""
    import urllib.request

    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)


def download_coco(data_dir: str = "/vol/data") -> tuple[str, str]:
    """Download full COCO dataset if not present.

    Returns:
        (train_path, val_path) tuple.
    """
    import shutil
    import zipfile
    from pathlib import Path

    data_path = Path(data_dir) / "coco"
    data_path.mkdir(parents=True, exist_ok=True)

    train_images = data_path / "images" / "train2017"
    val_images = data_path / "images" / "val2017"

    if train_images.exists() and val_images.exists() and (data_path / "labels").exists():
        n_train = len(list(train_images.glob("*.jpg")))
        n_val = len(list(val_images.glob("*.jpg")))
        print(f"COCO already exists: {n_train} train, {n_val} val")
        return str(train_images), str(val_images)

    base_url = "http://images.cocodataset.org/zips"

    for split in ["train2017", "val2017"]:
        dest = data_path / "images" / split
        if dest.exists() and len(list(dest.glob("*.jpg"))) > 0:
            print(f"{split} already exists")
            continue

        zip_path = data_path / f"{split}.zip"
        url = f"{base_url}/{split}.zip"
        download_with_progress(url, str(zip_path))

        print(f"Extracting {split}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path / "images")
        zip_path.unlink()

    if not (data_path / "labels" / "train2017").exists():
        labels_url = (
            "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
        )
        labels_zip = data_path / "coco2017labels.zip"
        download_with_progress(labels_url, str(labels_zip))

        with zipfile.ZipFile(labels_zip, "r") as zf:
            zf.extractall(data_path)
        labels_zip.unlink()

        extracted_labels = data_path / "coco" / "labels"
        if extracted_labels.exists():
            shutil.move(str(extracted_labels), str(data_path / "labels"))
            shutil.rmtree(data_path / "coco", ignore_errors=True)

    n_train = len(list(train_images.glob("*.jpg")))
    n_val = len(list(val_images.glob("*.jpg")))
    print(f"COCO ready: {n_train} train, {n_val} val")
    return str(train_images), str(val_images)


def convert_voc_xml_to_yolo(xml_path: str) -> list[str]:
    """Convert VOC XML annotation to YOLO format lines.
    
    Extracts image dimensions from XML <size> element instead of reading the image.
    """
    import xml.etree.ElementTree as ET

    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find("size")
    if size is None:
        return []
    img_w = int(size.find("width").text)  # type: ignore[union-attr]
    img_h = int(size.find("height").text)  # type: ignore[union-attr]
    
    lines = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text == "1":
            continue

        name = obj.find("name")
        if name is None or name.text not in classes:
            continue

        cls_id = classes.index(name.text)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)  # type: ignore[union-attr]
        ymin = float(bbox.find("ymin").text)  # type: ignore[union-attr]
        xmax = float(bbox.find("xmax").text)  # type: ignore[union-attr]
        ymax = float(bbox.find("ymax").text)  # type: ignore[union-attr]

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def download_voc(data_dir: str = "/vol/data") -> tuple[str, str]:
    """Download Pascal VOC 2007+2012 dataset and convert to YOLO format.

    Returns:
        (train_path, val_path) tuple.
    """
    import shutil
    import tarfile
    from pathlib import Path

    data_path = Path(data_dir) / "voc"
    data_path.mkdir(parents=True, exist_ok=True)

    train_images = data_path / "images" / "train"
    val_images = data_path / "images" / "val"
    train_labels = data_path / "labels" / "train"
    val_labels = data_path / "labels" / "val"

    if all(p.exists() for p in [train_images, val_images, train_labels, val_labels]):
        n_train = len(list(train_images.glob("*.jpg")))
        n_val = len(list(val_images.glob("*.jpg")))
        if n_train > 0 and n_val > 0:
            print(f"VOC already exists: {n_train} train, {n_val} val")
            return str(train_images), str(val_images)

    voc_path = data_path / "VOCdevkit"

    # Download and extract all VOC tarballs
    # Use marker files to track what's been extracted
    urls = [
        "http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
        "http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
        "http://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar",
    ]
    for url in urls:
        name = url.split("/")[-1]
        marker = data_path / f".{name}.done"
        if marker.exists():
            continue
        tar_path = data_path / name
        if not tar_path.exists():
            download_with_progress(url, str(tar_path))
        print(f"Extracting {name}...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_path)
        tar_path.unlink()
        marker.touch()

    for split in ["train", "val"]:
        (data_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    test_file = voc_path / "VOC2007" / "ImageSets" / "Main" / "test.txt"
    test_ids: set[str] = set()
    if test_file.exists():
        with open(test_file) as f:
            test_ids = {line.strip() for line in f if line.strip()}
    print(f"VOC test_file exists: {test_file.exists()}, test_ids count: {len(test_ids)}")

    # Collect all images first for progress bar
    all_images: list[tuple[Path, Path, Path]] = []
    for voc_year in ["VOC2007", "VOC2012"]:
        year_path = voc_path / voc_year
        if not year_path.exists():
            continue
        jpeg_dir = year_path / "JPEGImages"
        annot_dir = year_path / "Annotations"
        for img_file in jpeg_dir.glob("*.jpg"):
            all_images.append((img_file, annot_dir, year_path))

    for img_file, annot_dir, year_path in tqdm(all_images, desc="Converting VOC to YOLO"):
        img_id = img_file.stem
        xml_file = annot_dir / f"{img_id}.xml"

        is_val = img_id in test_ids and year_path.name == "VOC2007"
        dest_img_dir = val_images if is_val else train_images
        dest_lbl_dir = val_labels if is_val else train_labels

        dest_img = dest_img_dir / img_file.name
        if dest_img.exists():
            continue

        shutil.copy(str(img_file), str(dest_img))

        if xml_file.exists():
            lines = convert_voc_xml_to_yolo(str(xml_file))
            if lines:
                label_file = dest_lbl_dir / f"{img_id}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(lines))

    n_train = len(list(train_images.glob("*.jpg")))
    n_val = len(list(val_images.glob("*.jpg")))
    print(f"VOC ready: {n_train} train, {n_val} val")
    return str(train_images), str(val_images)


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
    dataset: str = "coco128",
) -> dict:
    """Train YOLO on COCO dataset.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_period: Validate every N epochs.
        use_pretrained: If True, start from pretrained weights.
        augment_preset: Augmentation preset ("full", "light", "minimal").
        dataset: Dataset to use ("coco128" or "coco").

    Returns:
        Training metrics.
    """
    import torch

    from yolo import YOLO, Trainer
    from yolo.data.config import AugmentConfig, DataConfig

    if dataset == "coco128":
        train_path = download_coco128(f"{VOLUME_PATH}/data")
        val_path = train_path
        num_classes = 80
    elif dataset == "coco":
        train_path, val_path = download_coco(f"{VOLUME_PATH}/data")
        num_classes = 80
    elif dataset == "voc":
        train_path, val_path = download_voc(f"{VOLUME_PATH}/data")
        num_classes = 20
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'coco128', 'coco', or 'voc'")

    model = YOLO.from_yaml("/root/configs/models/gelan-c.yaml", num_classes=num_classes)

    if use_pretrained and dataset != "voc":
        weights_file = download_and_convert_weights(f"{VOLUME_PATH}/weights")
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights")
    elif use_pretrained and dataset == "voc":
        print("Warning: Pretrained weights are for 80 classes, VOC has 20. Training from scratch.")

    augment = AugmentConfig.from_preset(augment_preset)  # type: ignore[arg-type]
    print(f"Using augmentation preset: {augment_preset}")
    print(f"  mosaic={augment.mosaic}, mixup={augment.mixup}")

    data = DataConfig(
        train_path=train_path,
        val_path=val_path,
        num_classes=num_classes,
        batch_size=batch_size,
        workers=16,
        augment=augment,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model,
        data,
        epochs=epochs,
        lr=lr,
        val_period=val_period,
        output_dir=f"{VOLUME_PATH}/runs",
        device="cuda",
        optimizer=optimizer,
    )

    trainer.train()
    volume.commit()

    return {
        "epochs": epochs,
        "best_map50": trainer.best_fitness,
        "checkpoint": f"{VOLUME_PATH}/runs/last.pt",
    }


@app.function(gpu="T4", volumes={VOLUME_PATH: volume}, timeout=7200)
def run_eval_pretrained() -> dict:
    return _eval_pretrained_impl()


# GPU must be set via environment variable because Modal decorators
# evaluate at import time, before CLI args are parsed.
GPU = os.environ.get("GPU", "T4")


@app.function(gpu=GPU, volumes={VOLUME_PATH: volume}, timeout=86400)
def train(
    epochs: int,
    batch_size: int,
    lr: float,
    val_period: int,
    use_pretrained: bool = False,
    augment_preset: str = "full",
    dataset: str = "coco128",
) -> dict:
    return _train_impl(
        epochs, batch_size, lr, val_period, use_pretrained, augment_preset, dataset
    )


@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.01,
    val_period: int = 5,
    dataset: str = "coco128",
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
        dataset: Dataset ("coco128" for smoke test, "coco" for full training).
        eval_pretrained: If True, evaluate pretrained weights instead of training.
        use_pretrained: If True, start training from pretrained weights.
        augment_preset: Augmentation preset ("full", "light", "minimal").

    Environment:
        GPU: GPU type (T4, L4, A10G, A100). Default: T4.
    """
    if eval_pretrained:
        print("Evaluating pretrained weights on COCO128...")
        result = run_eval_pretrained.remote()
        print(f"Evaluation complete: {result}")
        return

    if use_pretrained and lr == 0.01:
        lr = 0.001
        print(f"Auto-adjusted LR to {lr} for fine-tuning (override with --lr)")

    print(f"Dataset: {dataset}")
    print(f"Training: {epochs} epochs, batch {batch_size}, lr {lr}")
    print(f"Validation every {val_period} epochs, GPU: {GPU}")
    if use_pretrained:
        print("Starting from pretrained weights")

    result = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_period=val_period,
        use_pretrained=use_pretrained,
        augment_preset=augment_preset,
        dataset=dataset,
    )
    print(f"Training complete: {result}")
