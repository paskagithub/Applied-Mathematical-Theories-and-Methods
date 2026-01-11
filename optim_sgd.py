"""Mini-batch SGD training with optional learning-rate schedules."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics_part2 import full_batch_grad_norm


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _loss_with_l2(
    net: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    l2_reg: float,
) -> torch.Tensor:
    pred = net(X)
    loss = loss_fn(pred, y)
    if l2_reg > 0.0:
        l2_term = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for param in net.parameters():
            l2_term = l2_term + param.pow(2).sum()
        loss = loss + 0.5 * l2_reg * l2_term
    return loss


def _is_classification(y: torch.Tensor) -> bool:
    unique_vals = torch.unique(y.detach())
    if unique_vals.numel() > 2:
        return False
    return torch.all((unique_vals == -1) | (unique_vals == 1)).item()


def _accuracy_from_logits(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred_labels = torch.where(pred >= 0, torch.tensor(1, device=pred.device), torch.tensor(-1, device=pred.device))
    return float((pred_labels == y).float().mean().item())


def _apply_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_epochs: int,
    base_lr: float,
    schedule: str,
    step_gamma: float,
    step_every: int,
) -> None:
    if schedule == "step":
        if epoch > 1 and (epoch - 1) % step_every == 0:
            for group in optimizer.param_groups:
                group["lr"] *= step_gamma
    elif schedule == "cosine":
        if max_epochs <= 1:
            lr = base_lr
        else:
            lr_min = base_lr * 0.01
            progress = (epoch - 1) / (max_epochs - 1)
            lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")


def train_sgd(
    net: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: LossFn,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
    max_epochs: int = 60,
    batch_size: int = 128,
    lr: float = 0.1,
    schedule: str = "step",
    step_gamma: float = 0.1,
    step_every: int = 20,
    grad_clip: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train using mini-batch SGD with optional learning-rate schedule."""
    start_time = time.perf_counter()
    status = "OK"
    error_msg = ""
    final_train_loss = float("nan")
    final_test_loss = float("nan")
    final_train_acc = float("nan")
    final_test_acc = float("nan")
    final_grad_norm = float("nan")

    try:
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        is_classification = _is_classification(y_train)

        for epoch in range(1, max_epochs + 1):
            _apply_schedule(optimizer, epoch, max_epochs, lr, schedule, step_gamma, step_every)
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _loss_with_l2(net, xb, yb, loss_fn, l2_reg)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch}/{max_epochs}")

        net.eval()
        with torch.no_grad():
            train_pred = net(X_train)
            test_pred = net(X_test)
            final_train_loss = float(loss_fn(train_pred, y_train).item())
            final_test_loss = float(loss_fn(test_pred, y_test).item())
            if l2_reg > 0.0:
                l2_term = torch.tensor(0.0, device=train_pred.device, dtype=train_pred.dtype)
                for param in net.parameters():
                    l2_term = l2_term + param.pow(2).sum()
                final_train_loss = float(final_train_loss + 0.5 * l2_reg * l2_term.item())
            if is_classification:
                final_train_acc = _accuracy_from_logits(train_pred, y_train)
                final_test_acc = _accuracy_from_logits(test_pred, y_test)

        final_grad_norm = float(full_batch_grad_norm(net, X_train, y_train, loss_fn, l2_reg=l2_reg))
    except Exception as exc:  # pragma: no cover - defensive
        status = "ERROR"
        error_msg = str(exc)

    time_s = float(time.perf_counter() - start_time)

    return {
        "epochs_run": float(max_epochs),
        "time_s": float(time_s),
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "train_acc": float(final_train_acc),
        "test_acc": float(final_test_acc),
        "final_grad_norm": float(final_grad_norm),
        "status": status,
        "error_msg": error_msg,
    }
