"""Fit exp(x) sin(10x) on [0, 1] with a Cartan group-valued neural network.

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

    def __init__(self, width: int = 32, hidden_layers: int = 2) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        layers: list[nn.Module] = [HyperbolicEmbedding()]
        for layer_index in range(hidden_layers):
            input_width = 2 if layer_index == 0 else width
            layers.extend(
                [HyperbolicLinear(input_width, width), HyperbolicActivation(nn.Identity)]
            )
        layers.append(HyperbolicRegression(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def target_function(x: torch.Tensor) -> torch.Tensor:
    """Return sin(10x) exp(x) elementwise."""
    return torch.sin(10.0 * x) * torch.exp(x)


def train_model(
    epochs: int, learning_rate: float, hidden_layers: int, width: int
) -> tuple[CartanExpNetwork, list[tuple[int, float, float]]]:
    """Train on one grid and record losses on distinct train and validation grids."""
    x_train = torch.linspace(0.0, 1.0, 256).reshape(-1, 1)
    y_train = target_function(x_train)
    # Midpoint samples do not occur in the training grid.
    x_validation = (torch.arange(255, dtype=torch.float32) + 0.5).reshape(-1, 1) / 255.0
    y_validation = target_function(x_validation)

    model = CartanExpNetwork(width=width, hidden_layers=hidden_layers)
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: list[tuple[int, float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        prediction = model(x_train)
        train_loss = loss_fn(prediction, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss_value = loss_fn(model(x_train), y_train).item()
            validation_loss = loss_fn(model(x_validation), y_validation).item()
        history.append((epoch, train_loss_value, validation_loss))

        if epoch == 1 or epoch % max(epochs // 5, 1) == 0:
            print(
                f"epoch {epoch:>5}/{epochs}: "
                f"train_loss={train_loss_value:.6e}, "
                f"validation_loss={validation_loss:.6e}"
            )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-layers", type=int, choices=(2, 3, 5), default=2)
    parser.add_argument("--width", type=int, default=32)
    output_dir = os.path.join(os.path.dirname(__file__), "fit_exp_doc")
    os.makedirs(output_dir, exist_ok=True)
    parser.add_argument("--save", type=str, help="path for the fitted-function plot")
    parser.add_argument("--loss-save", type=str, help="CSV path for per-epoch losses")
    parser.add_argument("--loss-plot", type=str, help="path for the loss plot")
    args = parser.parse_args()
    prefix = "exp_fit" if args.hidden_layers == 2 else f"cartan_{args.hidden_layers}layer_exp_fit"
    if args.width != 32:
        prefix = f"cartan_width{args.width}_{args.hidden_layers}layer_exp_fit"
    args.save = args.save or os.path.join(output_dir, f"{prefix}_function.png")
    args.loss_save = args.loss_save or os.path.join(output_dir, f"{prefix}_losses.csv")
    args.loss_plot = args.loss_plot or os.path.join(output_dir, f"{prefix}_losses.png")

    torch.manual_seed(0)
    model, history = train_model(
        args.epochs, args.learning_rate, args.hidden_layers, args.width
    )
    with open(args.loss_save, "w", encoding="utf-8") as loss_file:
        loss_file.write("epoch,training_loss,validation_loss\n")
        for epoch, train_loss, validation_loss in history:
            loss_file.write(f"{epoch},{train_loss:.10e},{validation_loss:.10e}\n")
    print(f"loss history saved to {args.loss_save}")

    epochs, training_losses, validation_losses = zip(*history)
    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, training_losses, label="Training MSE")
    plt.semilogy(epochs, validation_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.loss_plot, dpi=150)
    plt.close()
    print(f"loss plot saved to {args.loss_plot}")

    x_plot = torch.linspace(0.0, 1.0, 500).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_plot).flatten()
    y_true = target_function(x_plot).flatten()
    max_error = torch.max(torch.abs(y_pred - y_true)).item()
    print(f"maximum absolute error on plot grid: {max_error:.6e}")

    x_values = x_plot.flatten().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_true.numpy(), label=r"$e^x\sin(10x)$", linewidth=2)
    plt.plot(x_values, y_pred.numpy(), "--", label="Cartan approximation", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title(r"Approximating $e^x\sin(10x)$ on $[0, 1]$ with a Cartan network")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.save, dpi=150)
    print(f"plot saved to {args.save}")


if __name__ == "__main__":
    main()
