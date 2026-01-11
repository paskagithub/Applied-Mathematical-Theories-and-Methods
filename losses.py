"""Loss functions for regression and binary classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def squared_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return mean 0.5 * (pred - y)^2."""
    return 0.5 * (pred - y).pow(2).mean()


def logistic_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return mean log(1 + exp(-y * pred)) for y in {-1, +1}."""
    return F.softplus(-y * pred).mean()


def accuracy_from_logits(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy from logits using sign(pred) vs y in {-1, +1}."""
    pred_labels = torch.where(pred >= 0, torch.tensor(1, device=pred.device), torch.tensor(-1, device=pred.device))
    return (pred_labels == y).float().mean().item()
