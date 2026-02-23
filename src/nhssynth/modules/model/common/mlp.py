from typing import Optional

import torch
import torch.nn as nn

from nhssynth.common.constants import ACTIVATION_FUNCTIONS


class _MLPBlock(nn.Module):
    """A single hidden layer: Linear → [BatchNorm] → Activation → [Dropout] with optional residual."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: type,
        batch_norm: bool,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.act = activation()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual and (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn(self.linear(x))))
        if self.residual:
            out = out + x
        return out


class MLP(nn.Module):
    """
    A configurable multi-layer perceptron used as the generator and critic backbone in the GAN.

    Args:
        n_units_in: Input dimensionality.
        n_units_out: Output dimensionality.
        n_layers_hidden: Number of hidden layers.
        n_units_hidden: Width of each hidden layer.
        activation: Hidden layer activation function name (key in ACTIVATION_FUNCTIONS).
        batch_norm: Whether to apply BatchNorm after each hidden linear layer.
        dropout: Dropout probability for hidden layers (0 = disabled).
        residual: Whether to add residual skip connections within hidden layers of equal width.
        lr: Learning rate for the Adam optimiser.
        opt_betas: (beta1, beta2) for the Adam optimiser.
        activation_out: Optional activation on the output layer (key in ACTIVATION_FUNCTIONS).
    """

    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 256,
        activation: str = "leaky_relu",
        batch_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        lr: float = 2e-4,
        opt_betas: tuple = (0.9, 0.999),
        activation_out: Optional[str] = None,
    ) -> None:
        super().__init__()

        act_fn = ACTIVATION_FUNCTIONS[activation]

        layers: list[nn.Module] = []
        # First hidden layer: maps from input dim → hidden dim
        layers.append(_MLPBlock(n_units_in, n_units_hidden, act_fn, batch_norm, dropout, residual))
        # Remaining hidden layers: hidden dim → hidden dim (residual possible)
        for _ in range(n_layers_hidden - 1):
            layers.append(_MLPBlock(n_units_hidden, n_units_hidden, act_fn, batch_norm, dropout, residual))
        # Output projection
        layers.append(nn.Linear(n_units_hidden, n_units_out))
        if activation_out is not None:
            layers.append(ACTIVATION_FUNCTIONS[activation_out]())

        self.network = nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, betas=opt_betas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
