"""Utilities for reproducible seeding and dtype helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int) -> None:
    """Set RNG seeds for python, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_dtype(dtype_str: str) -> "torch.dtype":
    """Return a torch dtype for the provided string."""
    if torch is None:
        raise RuntimeError("torch is not installed")
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def to_device(x: Any, device: str):
    """Move tensors to the specified device when possible."""
    if torch is None:
        return x
    if hasattr(x, "to"):
        return x.to(device)
    return x
