"""Device detection utilities."""

import torch


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        device: Device string. "auto" detects best available.
            Options: "auto", "cuda", "mps", "cpu", or specific like "cuda:0"

    Returns:
        torch.device for the selected device.
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
