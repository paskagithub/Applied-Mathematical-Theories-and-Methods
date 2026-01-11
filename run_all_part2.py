"""Run mandatory Part 2 experiments and write results CSVs."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

import data_mnist01
import data_synthetic
import losses
import optim_gd
import optim_kfac
import optim_sgd
import utils_seed
from model_shallow import ShallowNet, init_shallow_net

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch required for runs
    raise RuntimeError("torch is required to run Part 2 experiments") from exc


COLUMNS = [
    "run_name",
    "dataset",
    "optimizer",
    "activation",
    "m",
    "d",
    "seed",
    "dtype",
    "device",
    "N_train",
    "N_test",
    "l2_reg",
    "max_epochs",
    "batch_size",
    "lr",
    "schedule",
    "step_gamma",
    "step_every",
    "damping",
    "update_freq",
    "eta0",
    "beta",
    "c",
    "grad_tol",
    "patience",
    "min_delta",
    "epochs_run",
    "time_s",
    "final_train_loss",
    "final_test_loss",
    "train_acc",
    "test_acc",
    "final_grad_norm",
    "status",
    "error_msg",
]


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value.lower() == "none":
        return None
    return int(value)


def _select_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(d: int, m: int, activation: str, dtype: torch.dtype, device: str, seed: int) -> ShallowNet:
    net = ShallowNet(d=d, m=m, activation=activation, dtype=dtype)
    init_shallow_net(net, seed=seed)
    return net.to(device)


def _base_row(
    run_name: str,
    dataset: str,
    optimizer: str,
    activation: str,
    m: int,
    d: int,
    seed: int,
    dtype: str,
    device: str,
    N_train: int,
    N_test: int,
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "dataset": dataset,
        "optimizer": optimizer,
        "activation": activation,
        "m": m,
        "d": d,
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "N_train": N_train,
        "N_test": N_test,
        "l2_reg": math.nan,
        "max_epochs": math.nan,
        "batch_size": math.nan,
        "lr": math.nan,
        "schedule": None,
        "step_gamma": math.nan,
        "step_every": math.nan,
        "damping": math.nan,
        "update_freq": math.nan,
        "eta0": math.nan,
        "beta": math.nan,
        "c": math.nan,
        "grad_tol": math.nan,
        "patience": math.nan,
        "min_delta": math.nan,
        "epochs_run": math.nan,
        "time_s": math.nan,
        "final_train_loss": math.nan,
        "final_test_loss": math.nan,
        "train_acc": math.nan,
        "test_acc": math.nan,
        "final_grad_norm": math.nan,
        "status": "ERROR",
        "error_msg": "",
    }


def _merge_result(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    row.update(
        {
            "epochs_run": result.get("epochs_run", math.nan),
            "time_s": result.get("time_s", math.nan),
            "final_train_loss": result.get("final_train_loss", math.nan),
            "final_test_loss": result.get("final_test_loss", math.nan),
            "train_acc": result.get("train_acc", math.nan),
            "test_acc": result.get("test_acc", math.nan),
            "final_grad_norm": result.get("final_grad_norm", math.nan),
        }
    )
    status = result.get("status", "ERROR")
    if status == "ERROR":
        row["status"] = "ERROR"
        row["error_msg"] = result.get("error_msg", "")
    else:
        row["status"] = "OK"
        row["error_msg"] = ""
    return row


def _run_synthetic(
    seed: int,
    dtype: str,
    device: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    dataset = data_synthetic.make_synthetic_regression(seed=seed, dtype=dtype, device=device)
    X_train = dataset["X_train"]
    y_train = dataset["y_train"].squeeze(-1)
    X_test = dataset["X_test"]
    y_test = dataset["y_test"].squeeze(-1)
    meta = dataset["meta"]

    m = 256
    activation = "relu"
    torch_dtype = utils_seed.get_torch_dtype(dtype)

    rows: List[Dict[str, Any]] = []

    gd_row = _base_row(
        run_name="R_GD_ARMIJO",
        dataset="synthetic",
        optimizer="GD_ARMIJO",
        activation=activation,
        m=m,
        d=meta["d"],
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    gd_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 250,
            "eta0": 1.0,
            "beta": 0.5,
            "c": 1e-4,
            "grad_tol": 1e-4,
            "patience": 5,
            "min_delta": 1e-6,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(meta["d"], m, activation, torch_dtype, device, seed)
    result = optim_gd.train_gd_armijo(
        net,
        X_train,
        y_train,
        losses.squared_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=250,
        verbose=verbose,
    )
    rows.append(_merge_result(gd_row, result))

    sgd_row = _base_row(
        run_name="R_SGD",
        dataset="synthetic",
        optimizer="SGD",
        activation=activation,
        m=m,
        d=meta["d"],
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    sgd_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 60,
            "batch_size": 128,
            "lr": 0.1,
            "schedule": "step",
            "step_gamma": 0.1,
            "step_every": 20,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(meta["d"], m, activation, torch_dtype, device, seed)
    result = optim_sgd.train_sgd(
        net,
        X_train,
        y_train,
        losses.squared_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=60,
        batch_size=128,
        lr=0.1,
        schedule="step",
        step_gamma=0.1,
        step_every=20,
        verbose=verbose,
    )
    rows.append(_merge_result(sgd_row, result))

    kfac_row = _base_row(
        run_name="R_KFAC",
        dataset="synthetic",
        optimizer="KFAC",
        activation=activation,
        m=m,
        d=meta["d"],
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    kfac_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 35,
            "batch_size": 256,
            "lr": 0.2,
            "damping": 1e-2,
            "update_freq": 1,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(meta["d"], m, activation, torch_dtype, device, seed)
    result = optim_kfac.train_kfac(
        net,
        X_train,
        y_train,
        losses.squared_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=35,
        batch_size=256,
        lr=0.2,
        damping=1e-2,
        update_freq=1,
        verbose=verbose,
    )
    rows.append(_merge_result(kfac_row, result))

    return rows


def _run_mnist(
    seed: int,
    dtype: str,
    device: str,
    verbose: bool,
    mnist_train_max: Optional[int],
    mnist_test_max: Optional[int],
) -> List[Dict[str, Any]]:
    dataset = data_mnist01.load_mnist01(
        seed=seed,
        N_train_max=mnist_train_max,
        N_test_max=mnist_test_max,
        dtype=dtype,
        device=device,
    )

    rows: List[Dict[str, Any]] = []
    m = 256
    activation = "relu"
    torch_dtype = utils_seed.get_torch_dtype(dtype)

    base_meta = {
        "d": 784,
        "N_train": math.nan,
        "N_test": math.nan,
    }

    if dataset.get("status") == "NO_DATA":
        error_msg = dataset.get("error_msg", "")
        for run_name, optimizer in [
            ("M_GD_ARMIJO", "GD_ARMIJO"),
            ("M_SGD", "SGD"),
            ("M_KFAC", "KFAC"),
        ]:
            row = _base_row(
                run_name=run_name,
                dataset="mnist01",
                optimizer=optimizer,
                activation=activation,
                m=m,
                d=base_meta["d"],
                seed=seed,
                dtype=dtype,
                device=device,
                N_train=base_meta["N_train"],
                N_test=base_meta["N_test"],
            )
            row["status"] = "NO_DATA"
            row["error_msg"] = error_msg
            rows.append(row)
        return rows

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    meta = dataset["meta"]

    gd_row = _base_row(
        run_name="M_GD_ARMIJO",
        dataset="mnist01",
        optimizer="GD_ARMIJO",
        activation=activation,
        m=m,
        d=784,
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    gd_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 60,
            "eta0": 1.0,
            "beta": 0.5,
            "c": 1e-4,
            "grad_tol": 1e-4,
            "patience": 5,
            "min_delta": 1e-6,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(784, m, activation, torch_dtype, device, seed)
    result = optim_gd.train_gd_armijo(
        net,
        X_train,
        y_train,
        losses.logistic_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=60,
        verbose=verbose,
    )
    rows.append(_merge_result(gd_row, result))

    sgd_row = _base_row(
        run_name="M_SGD",
        dataset="mnist01",
        optimizer="SGD",
        activation=activation,
        m=m,
        d=784,
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    sgd_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 15,
            "batch_size": 128,
            "lr": 0.1,
            "schedule": "step",
            "step_gamma": 0.1,
            "step_every": 20,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(784, m, activation, torch_dtype, device, seed)
    result = optim_sgd.train_sgd(
        net,
        X_train,
        y_train,
        losses.logistic_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=15,
        batch_size=128,
        lr=0.1,
        schedule="step",
        step_gamma=0.1,
        step_every=20,
        verbose=verbose,
    )
    rows.append(_merge_result(sgd_row, result))

    kfac_row = _base_row(
        run_name="M_KFAC",
        dataset="mnist01",
        optimizer="KFAC",
        activation=activation,
        m=m,
        d=784,
        seed=seed,
        dtype=dtype,
        device=device,
        N_train=meta["N_train"],
        N_test=meta["N_test"],
    )
    kfac_row.update(
        {
            "l2_reg": 1e-4,
            "max_epochs": 10,
            "batch_size": 256,
            "lr": 0.2,
            "damping": 1e-2,
            "update_freq": 1,
        }
    )
    utils_seed.set_global_seed(seed)
    net = _build_model(784, m, activation, torch_dtype, device, seed)
    result = optim_kfac.train_kfac(
        net,
        X_train,
        y_train,
        losses.logistic_loss,
        X_test,
        y_test,
        l2_reg=1e-4,
        max_epochs=10,
        batch_size=256,
        lr=0.2,
        damping=1e-2,
        update_freq=1,
        verbose=verbose,
    )
    rows.append(_merge_result(kfac_row, result))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 2 experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="results_part2")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mnist_train_max", type=str, default=None)
    parser.add_argument("--mnist_test_max", type=str, default=None)
    args = parser.parse_args()

    mnist_train_max = _parse_optional_int(args.mnist_train_max)
    mnist_test_max = _parse_optional_int(args.mnist_test_max)

    device = _select_device(args.device)
    utils_seed.set_global_seed(args.seed)

    synthetic_rows = _run_synthetic(args.seed, args.dtype, device, args.verbose)
    mnist_rows = _run_mnist(args.seed, args.dtype, device, args.verbose, mnist_train_max, mnist_test_max)

    os.makedirs(args.out_dir, exist_ok=True)

    synthetic_df = pd.DataFrame(synthetic_rows, columns=COLUMNS)
    mnist_df = pd.DataFrame(mnist_rows, columns=COLUMNS)

    synthetic_path = os.path.join(args.out_dir, "results_part2_synthetic.csv")
    mnist_path = os.path.join(args.out_dir, "results_part2_mnist01.csv")

    synthetic_df.to_csv(synthetic_path, index=False)
    mnist_df.to_csv(mnist_path, index=False)


if __name__ == "__main__":
    main()
