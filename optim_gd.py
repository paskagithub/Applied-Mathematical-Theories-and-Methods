"""Full-batch gradient descent with Armijo backtracking line search."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterable, Tuple

import torch


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


def _grad_norm_sq(params: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += param.grad.pow(2).sum().item()
    return total


def _params_and_grads(net: torch.nn.Module) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    params = [p for p in net.parameters() if p.requires_grad]
    grads = [p.grad.detach().clone() for p in params]
    return params, grads


def _apply_step(params: Iterable[torch.Tensor], grads: Iterable[torch.Tensor], step: float) -> None:
    with torch.no_grad():
        for param, grad in zip(params, grads):
            param.add_(grad, alpha=-step)


def _restore_params(params: Iterable[torch.Tensor], originals: Iterable[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, orig in zip(params, originals):
            param.copy_(orig)


def _is_classification(y: torch.Tensor) -> bool:
    unique_vals = torch.unique(y.detach())
    if unique_vals.numel() > 2:
        return False
    return torch.all((unique_vals == -1) | (unique_vals == 1)).item()


def _accuracy_from_logits(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred_labels = torch.where(pred >= 0, torch.tensor(1, device=pred.device), torch.tensor(-1, device=pred.device))
    return float((pred_labels == y).float().mean().item())


def train_gd_armijo(
    net: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: LossFn,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
    max_epochs: int = 250,
    grad_tol: float = 1e-4,
    eta0: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    patience: int = 5,
    min_delta: float = 1e-6,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train using full-batch GD with Armijo backtracking line search."""
    start_time = time.perf_counter()
    status = "OK"
    error_msg = ""
    best_loss = float("inf")
    epochs_since_improve = 0
    final_train_loss = float("nan")
    final_test_loss = float("nan")
    final_train_acc = float("nan")
    final_test_acc = float("nan")
    final_grad_norm = float("nan")

    try:
        is_classification = _is_classification(y_train)
        for epoch in range(1, max_epochs + 1):
            net.train()
            net.zero_grad(set_to_none=True)
            loss = _loss_with_l2(net, X_train, y_train, loss_fn, l2_reg)
            loss.backward()
            grad_norm_sq = _grad_norm_sq(net.parameters())
            grad_norm = math.sqrt(grad_norm_sq)
            final_grad_norm = float(grad_norm)
            if grad_norm <= grad_tol:
                status = "GRAD_TOL"
                if verbose:
                    print(f"Early stop: grad norm {grad_norm:.3e} <= {grad_tol:.3e}")
                break

            params, grads = _params_and_grads(net)
            originals = [p.detach().clone() for p in params]
            base_loss = loss.detach().item()

            step = eta0
            accepted = False
            for _ in range(30):
                _restore_params(params, originals)
                _apply_step(params, grads, step)
                net.train()
                with torch.no_grad():
                    cand_loss = _loss_with_l2(net, X_train, y_train, loss_fn, l2_reg).item()
                if cand_loss <= base_loss - c * step * grad_norm_sq:
                    accepted = True
                    break
                step *= beta

            if not accepted:
                _restore_params(params, originals)
                _apply_step(params, grads, step)

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

            if final_train_loss < best_loss - min_delta:
                best_loss = final_train_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    status = "EARLY_STOP"
                    if verbose:
                        print("Early stop: no improvement")
                    break
    except Exception as exc:  # pragma: no cover - defensive
        status = "ERROR"
        error_msg = str(exc)

    time_s = float(time.perf_counter() - start_time)
    epochs_run = epoch if "epoch" in locals() else 0

    return {
        "epochs_run": float(epochs_run),
        "time_s": float(time_s),
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "train_acc": float(final_train_acc),
        "test_acc": float(final_test_acc),
        "final_grad_norm": float(final_grad_norm),
        "status": status,
        "error_msg": error_msg,
    }
