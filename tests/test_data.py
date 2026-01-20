"""Tests for data loading and augmentation."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from yolo.data import (
    HSV,
    Albumentations,
    AugmentConfig,
    CacheMode,
    Compose,
    DataConfig,
    Letterbox,
    NormalizeLabels,
    RandomFlip,
    Sample,
    YOLODataset,
    augment_hsv,
    collate_fn,
    letterbox,
    xywhn2xyxy,
    xyxy2xywhn,
)


class TestCoordinateConversions:
    """Tests for coordinate conversion functions."""

    def test_xywhn2xyxy_basic(self):
        """Test normalized xywh to xyxy conversion."""
        xywhn = np.array([[0.5, 0.5, 0.2, 0.2]])
        xyxy = xywhn2xyxy(xywhn, w=640, h=640)
        expected = np.array([[256, 256, 384, 384]])
        np.testing.assert_array_almost_equal(xyxy, expected)

    def test_xyxy2xywhn_basic(self):
        """Test xyxy to normalized xywh conversion."""
        xyxy = np.array([[256.0, 256.0, 384.0, 384.0]])
        xywhn = xyxy2xywhn(xyxy, w=640, h=640)
        expected = np.array([[0.5, 0.5, 0.2, 0.2]])
        np.testing.assert_array_almost_equal(xywhn, expected)

    def test_roundtrip(self):
        """Test xywhn -> xyxy -> xywhn roundtrip."""
        original = np.array([[0.3, 0.4, 0.15, 0.25]])
        xyxy = xywhn2xyxy(original, w=640, h=480)
        recovered = xyxy2xywhn(xyxy, w=640, h=480)
        np.testing.assert_array_almost_equal(original, recovered)


class TestLetterbox:
    """Tests for letterbox resizing."""

    def test_square_image(self):
        """Test letterboxing a square image."""
        img = np.zeros((480, 480, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, 640, auto=False)
        assert result.shape == (640, 640, 3)

    def test_wide_image(self):
        """Test letterboxing a wide image."""
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, 640, auto=False)
        assert result.shape == (640, 640, 3)

    def test_tall_image(self):
        """Test letterboxing a tall image."""
        img = np.zeros((640, 360, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, 640, auto=False)
        assert result.shape == (640, 640, 3)


class TestAugmentHSV:
    """Tests for HSV augmentation."""

    def test_no_change_with_zero_gains(self):
        """No change when gains are zero."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original = img.copy()
        augment_hsv(img, hgain=0, sgain=0, vgain=0)
        np.testing.assert_array_equal(img, original)

    def test_in_place_modification(self):
        """HSV augmentation modifies in place."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original = img.copy()
        augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5)
        assert not np.array_equal(img, original)

    def test_output_range(self):
        """HSV augmentation keeps values in valid range."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5)
        assert img.min() >= 0
        assert img.max() <= 255


class TestTransforms:
    """Tests for composable transforms."""

    def test_compose(self):
        """Test Compose chains transforms."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]])
        sample = Sample(img=img, labels=labels, img_size=640, original_shape=(480, 640))

        transforms = Compose([
            Letterbox(scaleup=False),
            NormalizeLabels(),
        ])

        result = transforms(sample)
        assert result.img.shape[0] == 640
        assert result.img.shape[1] == 640

    def test_hsv_transform(self):
        """Test HSV transform."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        labels = np.zeros((0, 5))
        sample = Sample(img=img.copy(), labels=labels, img_size=640, original_shape=(640, 640))

        transform = HSV(h=0.5, s=0.5, v=0.5)
        result = transform(sample)

        assert not np.array_equal(result.img, img)

    def test_random_flip(self):
        """Test RandomFlip transform."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:, :320] = 255  # Left half white
        labels = np.array([[0, 0.25, 0.5, 0.5, 1.0]])  # Normalized xywh
        sample = Sample(
            img=img.copy(), labels=labels.copy(), img_size=640, original_shape=(640, 640)
        )

        transform = RandomFlip(p_horizontal=1.0, p_vertical=0.0)
        result = transform(sample)

        assert result.img[:, 320:].mean() == 255  # Right half now white
        assert abs(result.labels[0, 1] - 0.75) < 1e-5  # x flipped

    def test_letterbox_transform(self):
        """Test Letterbox transform."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]])
        sample = Sample(img=img, labels=labels, img_size=640, original_shape=(480, 640))

        transform = Letterbox(scaleup=False)
        result = transform(sample)

        assert result.img.shape == (640, 640, 3)

    def test_albumentations_transform(self):
        """Test Albumentations wrapper transform."""
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        # Labels in normalized xywh format (what Albumentations expects)
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]])
        sample = Sample(img=img, labels=labels, img_size=640, original_shape=(640, 640))

        # Use high probability to ensure transforms are applied
        transform = Albumentations(blur_p=1.0, median_blur_p=0.0, to_gray_p=0.0, clahe_p=0.0)
        result = transform(sample)

        # Image should be modified (blurred)
        assert result.img.shape == (640, 640, 3)
        # Labels should be preserved
        assert len(result.labels) == 1
        np.testing.assert_array_almost_equal(result.labels[0], labels[0])

    def test_albumentations_empty_labels(self):
        """Test Albumentations handles empty labels."""
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        labels = np.zeros((0, 5))
        sample = Sample(img=img, labels=labels, img_size=640, original_shape=(640, 640))

        transform = Albumentations()
        result = transform(sample)

        assert result.img.shape == (640, 640, 3)
        assert result.labels.shape == (0, 5)


class TestYOLODataset:
    """Tests for YOLODataset."""

    def test_dataset_with_temp_data(self):
        """Test dataset loading with temporary test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            label_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            with open(label_dir / "test.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")
                f.write("1 0.3 0.3 0.1 0.1\n")

            transforms = Compose([Letterbox(), NormalizeLabels()])
            dataset = YOLODataset(img_dir, img_size=640, transforms=transforms)

            assert len(dataset) == 1
            img_tensor, labels, path, shapes = dataset[0]

            assert img_tensor.shape == (3, 640, 640)
            assert img_tensor.dtype == torch.float32
            assert img_tensor.min() >= 0
            assert img_tensor.max() <= 1

            assert labels.shape[1] == 6
            assert labels.shape[0] == 2

    def test_empty_labels(self):
        """Test dataset handles missing labels gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            transforms = Compose([Letterbox(), NormalizeLabels()])
            dataset = YOLODataset(img_dir, img_size=640, transforms=transforms)
            img_tensor, labels, path, shapes = dataset[0]

            assert labels.shape == (0, 6)

    def test_no_transforms(self):
        """Test dataset works without transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            label_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            with open(label_dir / "test.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

            dataset = YOLODataset(img_dir, img_size=640, transforms=None)
            img_tensor, labels, path, shapes = dataset[0]

            assert img_tensor.shape[0] == 3
            assert labels.shape == (1, 6)


class TestCollateFn:
    """Tests for collate function."""

    def test_collate_variable_targets(self):
        """Test collation of batches with variable targets."""
        batch = [
            (torch.randn(3, 640, 640), torch.zeros(2, 6), "img1.jpg", (480, 640)),
            (torch.randn(3, 640, 640), torch.zeros(5, 6), "img2.jpg", (480, 640)),
            (torch.randn(3, 640, 640), torch.zeros(0, 6), "img3.jpg", (480, 640)),
        ]

        imgs, labels, paths, shapes = collate_fn(batch)

        assert imgs.shape == (3, 3, 640, 640)
        assert labels.shape == (7, 6)
        assert len(paths) == 3

        assert (labels[:2, 0] == 0).all()
        assert (labels[2:7, 0] == 1).all()


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig(train_path="/path/to/train")
        assert config.img_size == 640
        assert config.batch_size == 16
        assert config.num_classes == 80

    def test_augment_config(self):
        """Test augmentation config defaults."""
        aug = AugmentConfig()
        assert aug.mosaic == 1.0
        assert aug.fliplr == 0.5
        assert aug.flipud == 0.0

    def test_path_conversion(self):
        """Test path auto-conversion to Path object."""
        config = DataConfig(train_path="/path/to/train", val_path="/path/to/val")
        assert isinstance(config.train_path, Path)
        assert isinstance(config.val_path, Path)

    def test_cache_mode_default(self):
        """Test CacheMode default is NONE."""
        config = DataConfig(train_path="/path/to/train")
        assert config.cache == CacheMode.NONE

    def test_augment_config_defaults_match_reference(self):
        """Test augmentation defaults match reference YOLOv9."""
        aug = AugmentConfig()
        assert aug.scale == 0.9  # Reference default
        assert aug.mixup == 0.15  # Reference default


class TestCacheMode:
    """Tests for CacheMode enum and caching behavior."""

    def test_cache_mode_values(self):
        """Test CacheMode enum values."""
        assert CacheMode.NONE.value == "none"
        assert CacheMode.RAM.value == "ram"
        assert CacheMode.DISK.value == "disk"

    def test_ram_caching(self):
        """Test RAM caching loads images into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            dataset = YOLODataset(img_dir, img_size=640, cache=CacheMode.RAM)

            # Image should be cached in RAM
            assert dataset.imgs[0] is not None
            assert dataset.imgs[0].shape == (480, 640, 3)

    def test_disk_caching(self):
        """Test disk caching creates .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            YOLODataset(img_dir, img_size=640, cache=CacheMode.DISK)

            # .npy file should exist
            npy_file = img_dir / "test.npy"
            assert npy_file.exists()

            # Should be able to load from .npy
            cached_img = np.load(npy_file)
            assert cached_img.shape[2] == 3  # Should be resized image

    def test_no_caching(self):
        """Test no caching leaves imgs list as None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            dataset = YOLODataset(img_dir, img_size=640, cache=CacheMode.NONE)

            # Image should not be cached
            assert dataset.imgs[0] is None


class TestRectTraining:
    """Tests for rectangular training."""

    def test_rect_sorts_by_aspect_ratio(self):
        """Test rect training sorts images by aspect ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            label_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            # Create images with different aspect ratios
            # Wide image (ar < 1)
            wide_img = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "wide.jpg"), wide_img)
            with open(label_dir / "wide.txt", "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

            # Tall image (ar > 1)
            tall_img = np.random.randint(0, 256, (600, 300, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "tall.jpg"), tall_img)
            with open(label_dir / "tall.txt", "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

            # Square image (ar = 1)
            square_img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "square.jpg"), square_img)
            with open(label_dir / "square.txt", "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")

            dataset = YOLODataset(
                img_dir, img_size=640, rect=True, batch_size=3, stride=32
            )

            # Should be sorted by aspect ratio (h/w)
            # wide (0.5) < square (1.0) < tall (2.0)
            assert "wide" in str(dataset.im_files[0])
            assert "square" in str(dataset.im_files[1])
            assert "tall" in str(dataset.im_files[2])

    def test_rect_computes_batch_shapes(self):
        """Test rect training computes batch shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            # Create a few images
            for i in range(4):
                img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)

            dataset = YOLODataset(
                img_dir, img_size=640, rect=True, batch_size=2, stride=32
            )

            assert dataset.batch is not None
            assert dataset.batch_shapes is not None
            assert len(dataset.batch_shapes) == 2  # 4 images / batch_size 2 = 2 batches

    def test_rect_disabled_by_default(self):
        """Test rect training is disabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            img_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)

            dataset = YOLODataset(img_dir, img_size=640)

            assert dataset.batch is None
            assert dataset.batch_shapes is None


class TestWorkerSeeding:
    """Tests for worker seeding and reproducibility."""

    def test_seed_worker_function_exists(self):
        """Test seed_worker function is importable."""
        from yolo.data.dataset import seed_worker

        # Should be callable
        assert callable(seed_worker)

    def test_create_dataloader_uses_generator(self):
        """Test create_dataloader sets up generator for reproducibility."""
        from yolo.data.dataset import create_dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            label_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "test.jpg"), img)
            with open(label_dir / "test.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

            config = DataConfig(train_path=img_dir, batch_size=1, workers=0)
            dataloader = create_dataloader(config, train=True)

            # DataLoader should have worker_init_fn set
            assert dataloader.worker_init_fn is not None
