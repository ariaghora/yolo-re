"""Visualize training augmentations."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yolo.data.dataset import YOLODataset
from yolo.data.transforms import default_train_transforms, default_val_transforms


def visualize_samples(dataset: YOLODataset, output_dir: Path, prefix: str, n: int = 5):
    """Save visualization of dataset samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(n, len(dataset))):
        img_tensor, labels, path, shape = dataset[i]
        
        # Convert back to BGR numpy for visualization
        img = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB -> BGR, denormalize
        img = np.ascontiguousarray(img)
        
        # Draw boxes (labels are [batch_idx, class, x, y, w, h] normalized)
        h, w = img.shape[:2]
        for lb in labels:
            cls, cx, cy, bw, bh = lb[1:].numpy()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"{int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        out_path = output_dir / f"{prefix}_{i}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path}")


def main():
    data_path = Path("data/coco128/images/train")
    output_dir = Path("debug_augment")
    
    # Dataset with training augmentations
    train_dataset = YOLODataset(
        path=data_path,
        img_size=640,
        transforms=None,
    )
    train_dataset.transforms = default_train_transforms(train_dataset)
    
    # Dataset with validation transforms (no augmentation)
    val_dataset = YOLODataset(
        path=data_path,
        img_size=640,
        transforms=default_val_transforms(),
    )
    
    print("Saving training augmented samples...")
    visualize_samples(train_dataset, output_dir, "train", n=10)
    
    print("\nSaving validation samples (no augmentation)...")
    visualize_samples(val_dataset, output_dir, "val", n=5)
    
    print(f"\nDone! Check {output_dir}/")


if __name__ == "__main__":
    main()
