"""Fit exp(x) on [0, 1] with a Cartan group-valued neural network.

Run interactively with:

    python fit_exp_cnn.py

For a non-interactive run, save the plot instead:

    python fit_exp_cnn.py --save exp_fit.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from layers import (  # noqa: E402
    HyperbolicActivation,
    HyperbolicEmbedding,
    HyperbolicLinear,
    HyperbolicRegression,
)
from geoopt.optim import RiemannianAdam  # noqa: E402


class CartanExpNetwork(nn.Module):
    """A group-valued network for a scalar input and scalar output.

    The embedding represents the input as a Cartan coordinate plus a fiber
    coordinate. Hidden layers operate on that representation using the
    algebra-aware layers from this repository.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            HyperbolicEmbedding(),
            HyperbolicLinear(2, width),
            HyperbolicActivation(nn.Identity),
            HyperbolicLinear(width, width),
            HyperbolicActivation(nn.Identity),
            HyperbolicRegression(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_model(epochs: int, learning_rate: float) -> CartanExpNetwork:
    """Train the Cartan network on samples of exp(x) for x in [0, 1]."""
    x_train = torch.linspace(0.0, 1.0, 256).reshape(-1, 1)
    y_train = torch.exp(x_train)

    model = CartanExpNetwork()
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % max(epochs // 5, 1) == 0:
            print(f"epoch {epoch:>5}/{epochs}: loss={loss.item():.6e}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--save", type=str, help="save the plot instead of displaying it")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = train_model(args.epochs, args.learning_rate)

    x_plot = torch.linspace(0.0, 1.0, 500).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_plot).flatten()
    y_true = torch.exp(x_plot).flatten()
    max_error = torch.max(torch.abs(y_pred - y_true)).item()
    print(f"maximum absolute error on plot grid: {max_error:.6e}")

    x_values = x_plot.flatten().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_true.numpy(), label=r"$e^x$", linewidth=2)
    plt.plot(x_values, y_pred.numpy(), "--", label="Cartan approximation", linewidth=2)
    plt.scatter([0, 1], [1, torch.e], color="black", zorder=3, label="endpoints")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title(r"Approximating $e^x$ on $[0, 1]$ with a Cartan network")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
