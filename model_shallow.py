"""Two-layer shallow neural network with scalar output."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import nn


class ShallowNet(nn.Module):
    """Shallow network: f(x) = (1/sqrt(m)) * v^T sigma(Wx)."""

    def __init__(
        self,
        d: int,
        m: int,
        activation: str = "relu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.activation_name = activation
        self.W = nn.Parameter(torch.empty(m, d, dtype=dtype))
        self.v = nn.Parameter(torch.empty(m, dtype=dtype))
        self._activation = self._get_activation(activation)

    @staticmethod
    def _get_activation(name: str):
        if name == "relu":
            return torch.relu
        if name == "tanh":
            return torch.tanh
        if name == "sigmoid":
            return torch.sigmoid
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N802
        hidden = self._activation(X @ self.W.t())
        return (hidden @ self.v) / math.sqrt(self.m)

    def flatten_parameters(self) -> torch.Tensor:
        """Return all parameters flattened into a single vector."""
        params: Iterable[torch.Tensor] = (self.W, self.v)
        return torch.cat([p.reshape(-1) for p in params])


def init_shallow_net(net: ShallowNet, seed: int, scheme: Optional[str] = None) -> None:
    """Initialize network parameters based on activation and scheme."""
    torch.manual_seed(seed)
    scheme_name = scheme or net.activation_name
    if scheme_name == "relu":
        nn.init.kaiming_normal_(net.W, nonlinearity="relu")
    elif scheme_name in {"tanh", "sigmoid"}:
        nn.init.xavier_normal_(net.W)
    else:
        raise ValueError(f"Unsupported init scheme: {scheme_name}")
    nn.init.normal_(net.v, mean=0.0, std=0.01)
