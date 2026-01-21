#!/usr/bin/env python3
"""Download COCO128 dataset for smoke testing."""

import shutil
import urllib.request
import zipfile
from pathlib import Path


def download_coco128(data_dir: Path | str = "data") -> Path:
    """Download and extract COCO128 dataset.

    Args:
        data_dir: Directory to store the dataset.

    Returns:
        Path to the extracted dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "coco128"
    if dataset_path.exists():
        print(f"COCO128 already exists at {dataset_path}")
        return dataset_path

    url = "https://ultralytics.com/assets/coco128.zip"
    zip_path = data_dir / "coco128.zip"

    print(f"Downloading COCO128 from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()

    # Reorganize to our expected structure:
    # data/coco128/
    #   images/
    #     train/  (symlink to train2017)
    #   labels/
    #     train/  (symlink to train2017)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    # Create train symlinks if they don't exist
    train_images = images_dir / "train"
    train_labels = labels_dir / "train"

    if not train_images.exists() and (images_dir / "train2017").exists():
        shutil.move(images_dir / "train2017", train_images)

    if not train_labels.exists() and (labels_dir / "train2017").exists():
        shutil.move(labels_dir / "train2017", train_labels)

    print(f"COCO128 ready at {dataset_path}")
    print(f"  Images: {len(list((train_images).glob('*.jpg')))} files")
    print(f"  Labels: {len(list((train_labels).glob('*.txt')))} files")

    return dataset_path


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    download_coco128(data_dir)
