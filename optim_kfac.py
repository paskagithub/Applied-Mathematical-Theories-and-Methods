"""Simplified KFAC optimizer for a shallow network."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics_part2 import full_batch_grad_norm


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _activation_derivative(name: str, z: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return (z > 0).to(dtype=z.dtype)
    if name == "tanh":
        return 1.0 - torch.tanh(z).pow(2)
    if name == "sigmoid":
        sig = torch.sigmoid(z)
        return sig * (1.0 - sig)
    raise ValueError(f"Unsupported activation: {name}")


def _is_classification(y: torch.Tensor) -> bool:
    unique_vals = torch.unique(y.detach())
    if unique_vals.numel() > 2:
        return False
    return torch.all((unique_vals == -1) | (unique_vals == 1)).item()


def _accuracy_from_logits(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred_labels = torch.where(pred >= 0, torch.tensor(1, device=pred.device), torch.tensor(-1, device=pred.device))
    return float((pred_labels == y).float().mean().item())


def _loss_with_l2(
    pred: torch.Tensor,
    y: torch.Tensor,
    net: torch.nn.Module,
    loss_fn: LossFn,
    l2_reg: float,
) -> torch.Tensor:
    loss = loss_fn(pred, y)
    if l2_reg > 0.0:
        l2_term = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for param in net.parameters():
            l2_term = l2_term + param.pow(2).sum()
        loss = loss + 0.5 * l2_reg * l2_term
    return loss


def train_kfac(
    net: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: LossFn,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
    max_epochs: int = 35,
    batch_size: int = 256,
    lr: float = 0.2,
    damping: float = 1e-2,
    update_freq: int = 1,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train using a simplified KFAC optimizer for a shallow network."""
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
        is_classification = _is_classification(y_train)
        activation_name = getattr(net, "activation_name", "relu")

        A_inv = None
        G_inv = None
        step_idx = 0

        for epoch in range(1, max_epochs + 1):
            net.train()
            for xb, yb in loader:
                step_idx += 1
                xb = xb.to(dtype=torch.float64)
                yb = yb.to(dtype=torch.float64)

                W = net.W
                v = net.v
                z = xb @ W.t()
                a = net._activation(z)  # type: ignore[attr-defined]
                pred = (a @ v) / math.sqrt(net.m)

                loss = _loss_with_l2(pred, yb, net, loss_fn, l2_reg)
                g_out = torch.autograd.grad(loss, pred, retain_graph=True)[0]
                sigma_prime = _activation_derivative(activation_name, z)
                g_z = (g_out.unsqueeze(1) * v.unsqueeze(0) / math.sqrt(net.m)) * sigma_prime

                grad_W = (g_z.t() @ xb) / xb.shape[0]
                grad_v = (a.t() @ g_out) / math.sqrt(net.m)
                if l2_reg > 0.0:
                    grad_W = grad_W + l2_reg * W
                    grad_v = grad_v + l2_reg * v

                if step_idx % update_freq == 0 or A_inv is None or G_inv is None:
                    A = (xb.t() @ xb) / xb.shape[0]
                    G = (g_z.t() @ g_z) / xb.shape[0]
                    eye_A = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                    eye_G = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                    try:
                        A_inv = torch.linalg.solve(A + damping * eye_A, eye_A)
                        G_inv = torch.linalg.solve(G + damping * eye_G, eye_G)
                    except RuntimeError as exc:
                        status = "ERROR"
                        error_msg = f"KFAC solve failed: {exc}"
                        raise

                grad_W_pre = G_inv @ grad_W @ A_inv

                with torch.no_grad():
                    W.add_(grad_W_pre, alpha=-lr)
                    v.add_(grad_v, alpha=-lr)

            if verbose:
                print(f"Epoch {epoch}/{max_epochs}")

        net.eval()
        with torch.no_grad():
            train_pred = net(X_train)
            test_pred = net(X_test)
            final_train_loss = float(_loss_with_l2(train_pred, y_train, net, loss_fn, l2_reg).item())
            final_test_loss = float(_loss_with_l2(test_pred, y_test, net, loss_fn, l2_reg).item())
            if is_classification:
                final_train_acc = _accuracy_from_logits(train_pred, y_train)
                final_test_acc = _accuracy_from_logits(test_pred, y_test)

        final_grad_norm = float(full_batch_grad_norm(net, X_train, y_train, loss_fn, l2_reg=l2_reg))
    except Exception as exc:  # pragma: no cover - defensive
        if status != "ERROR":
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
