"""Synthetic regression dataset generator (Problem R)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch optional at import
    torch = None  # type: ignore[assignment]
    _torch_import_error = exc
else:
    _torch_import_error = None


@dataclass(frozen=True)
class SyntheticMeta:
    d: int
    N_train: int
    N_test: int
    m_teacher: int
    noise_std: float
    dtype: str
    device: str


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for synthetic data generation") from _torch_import_error


def _torch_dtype(dtype_str: str) -> "torch.dtype":
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def make_synthetic_regression(
    seed: int,
    N_train: int = 5000,
    N_test: int = 1000,
    d: int = 20,
    m_teacher: int = 64,
    noise_std: float = 0.05,
    dtype: str = "float64",
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """Generate synthetic regression data following the Problem R setup."""
    _require_torch()

    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal(size=(N_train, d))
    X_test = rng.standard_normal(size=(N_test, d))

    W1 = rng.standard_normal(size=(m_teacher, d)) / np.sqrt(d)
    b1 = rng.standard_normal(size=(m_teacher,))
    W2 = rng.standard_normal(size=(m_teacher,)) / np.sqrt(m_teacher)

    def teacher(x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, x @ W1.T + b1)
        return hidden @ W2

    y_train = teacher(X_train) + rng.normal(scale=noise_std, size=N_train)
    y_test = teacher(X_test) + rng.normal(scale=noise_std, size=N_test)

    torch_dtype = _torch_dtype(dtype)
    X_train_t = torch.as_tensor(X_train, dtype=torch_dtype, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch_dtype, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch_dtype, device=device).unsqueeze(-1)
    y_test_t = torch.as_tensor(y_test, dtype=torch_dtype, device=device).unsqueeze(-1)

    meta = SyntheticMeta(
        d=d,
        N_train=N_train,
        N_test=N_test,
        m_teacher=m_teacher,
        noise_std=noise_std,
        dtype=dtype,
        device=device,
    )

    return {
        "task": "synthetic",
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "meta": asdict(meta),
    }
