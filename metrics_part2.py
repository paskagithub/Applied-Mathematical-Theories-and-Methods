"""Evaluation metrics for Part 2 experiments."""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def eval_regression(net: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: LossFn) -> Dict[str, float]:
    """Evaluate regression loss; accuracy is NaN."""
    net.eval()
    with torch.no_grad():
        pred = net(X)
        loss = loss_fn(pred, y).item()
    return {"loss": float(loss), "acc": float("nan")}


def eval_classification(
    net: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
) -> Dict[str, float]:
    """Evaluate classification loss and accuracy."""
    net.eval()
    with torch.no_grad():
        pred = net(X)
        loss = loss_fn(pred, y).item()
        pred_labels = torch.where(pred >= 0, torch.tensor(1, device=pred.device), torch.tensor(-1, device=pred.device))
        acc = (pred_labels == y).float().mean().item()
    return {"loss": float(loss), "acc": float(acc)}


def full_batch_grad_norm(
    net: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    l2_reg: float = 0.0,
) -> float:
    """Compute full-batch gradient norm with optional L2 regularization."""
    net.train()
    net.zero_grad(set_to_none=True)
    pred = net(X)
    loss = loss_fn(pred, y)
    if l2_reg > 0.0:
        l2_term = torch.tensor(0.0, device=pred.device)
        for param in net.parameters():
            l2_term = l2_term + param.pow(2).sum()
        loss = loss + 0.5 * l2_reg * l2_term
    loss.backward()
    grad_sq_sum = 0.0
    for param in net.parameters():
        if param.grad is not None:
            grad_sq_sum += param.grad.pow(2).sum().item()
    return float(math.sqrt(grad_sq_sum))
