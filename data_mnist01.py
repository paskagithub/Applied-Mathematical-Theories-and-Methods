"""MNIST 0/1 binary classification loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch optional at import
    torch = None  # type: ignore[assignment]
    _torch_import_error = exc
else:
    _torch_import_error = None

try:
    from torchvision.datasets import MNIST
except Exception as exc:  # pragma: no cover - torchvision optional at import
    MNIST = None  # type: ignore[assignment]
    _torchvision_import_error = exc
else:
    _torchvision_import_error = None


@dataclass(frozen=True)
class MNIST01Meta:
    N_train: int
    N_test: int
    dtype: str
    device: str
    N_train_max: Optional[int]
    N_test_max: Optional[int]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required to load MNIST") from _torch_import_error


def _torch_dtype(dtype_str: str) -> "torch.dtype":
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _load_mnist_split(train: bool):
    if MNIST is None:
        raise RuntimeError("torchvision is required to load MNIST") from _torchvision_import_error
    return MNIST(root="./data", train=train, download=True)


def _prepare_split(data: "torch.Tensor", targets: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    mask = (targets == 0) | (targets == 1)
    data = data[mask]
    targets = targets[mask]
    data = data.float().div(255.0).view(data.shape[0], -1)
    targets = torch.where(targets == 0, torch.tensor(-1), torch.tensor(1))
    return data, targets


def _shuffle_and_truncate(
    data: "torch.Tensor",
    targets: "torch.Tensor",
    seed: int,
    max_count: Optional[int],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.shape[0], generator=generator)
    data = data[perm]
    targets = targets[perm]
    if max_count is not None:
        data = data[:max_count]
        targets = targets[:max_count]
    return data, targets


def load_mnist01(
    seed: int,
    N_train_max: Optional[int] = None,
    N_test_max: Optional[int] = None,
    dtype: str = "float64",
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """Load MNIST digits {0,1} for binary classification."""
    try:
        _require_torch()
        train_ds = _load_mnist_split(train=True)
        test_ds = _load_mnist_split(train=False)
    except Exception as exc:
        return {
            "task": "mnist01",
            "status": "NO_DATA",
            "error_msg": str(exc),
            "X_train": None,
            "y_train": None,
            "X_test": None,
            "y_test": None,
            "meta": None,
        }

    X_train, y_train = _prepare_split(train_ds.data, train_ds.targets)
    X_test, y_test = _prepare_split(test_ds.data, test_ds.targets)

    X_train, y_train = _shuffle_and_truncate(X_train, y_train, seed, N_train_max)
    X_test, y_test = _shuffle_and_truncate(X_test, y_test, seed + 1, N_test_max)

    torch_dtype = _torch_dtype(dtype)
    X_train = X_train.to(device=device, dtype=torch_dtype)
    y_train = y_train.to(device=device, dtype=torch_dtype)
    X_test = X_test.to(device=device, dtype=torch_dtype)
    y_test = y_test.to(device=device, dtype=torch_dtype)

    meta = MNIST01Meta(
        N_train=X_train.shape[0],
        N_test=X_test.shape[0],
        dtype=dtype,
        device=device,
        N_train_max=N_train_max,
        N_test_max=N_test_max,
    )

    return {
        "task": "mnist01",
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "meta": asdict(meta),
    }
