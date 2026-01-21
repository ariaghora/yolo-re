"""Utilities for YOLO."""

from yolo.utils.device import get_device
from yolo.utils.nms import non_max_suppression

__all__ = ["get_device", "non_max_suppression"]
