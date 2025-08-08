from typing import Any, Callable, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nhssynth.common.constants import ACTIVATION_FUNCTIONS


class MultiActivationHead(nn.Module):
    """Final layer with multiple activations. Useful for tabular data."""

    def __init__(
        self,
        activations: list[tuple[nn.Module, int]],
    ) -> None:
        super(MultiActivationHead, self).__init__()
        self.activations = []
        self.activation_lengths = []

        for activation, length in activations:
            self.activations.append(activation)
            self.activation_lengths.append(length)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != np.sum(self.activation_lengths):
            raise RuntimeError(
                f"Shape mismatch for the activations: expected {np.sum(self.activation_lengths)}. Got shape {X.shape}."
            )

        split = 0
        out = torch.zeros(X.shape)

        for activation, step in zip(self.activations, self.activation_lengths):
            out[..., split : split + step] = activation(X[..., split : split + step])
            split += step

        return out


def _forward_skip_connection(self: nn.Module, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    X = X.float()
    out = self._forward(X, *args, **kwargs)
    return torch.cat([out, X], dim=-1)


def SkipConnection(cls: Type[nn.Module]) -> Type[nn.Module]:
    """Wraps a model to add a skip connection from the input to the output.

    Example:
    >>> ResidualBlock = SkipConnection(MLP)
    >>> res_block = ResidualBlock(n_units_in=10, n_units_out=3, n_units_hidden=64)
    >>> res_block(torch.ones(10, 10)).shape
    (10, 13)
    """

    class Wrapper(cls):
        pass

    Wrapper._forward = cls.forward
    Wrapper.forward = _forward_skip_connection
    Wrapper.__name__ = f"SkipConnection({cls.__name__})"
    Wrapper.__qualname__ = f"SkipConnection({cls.__qualname__})"
    Wrapper.__doc__ = f"""(With skipped connection) {cls.__doc__}"""
    return Wrapper


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0,
        batch_norm: bool = False,
        activation: Optional[Callable] = "relu",
    ) -> None:
        super(LinearLayer, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_units_in, n_units_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units_out))

        if activation is not None:
            layers.append(activation())

        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float())


ResidualLayer = SkipConnection(LinearLayer)


class MLP(nn.Module):
    """
    Fully connected or residual neural nets for classification and regression.

    Parameters
    ----------
    task_type: str
        classification or regression
    n_units_int: int
        Number of features
    n_units_out: int
        Number of outputs
    n_layers_hidden: int
        Number of hidden layers
    n_units_hidden: int
        Number of hidden units in each layer
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu', 'tanh' or 'leaky_relu'.
    lr: float
        learning rate for optimizer.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    random_state: int
        random_state used
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    dropout: float
        Dropout value. If 0, the dropout is not used.
    clipping_value: int, default 1
        Gradients clipping value
    batch_norm: bool
        Enable/disable batch norm
    early_stopping: bool
        Enable/disable early stopping
    residual: bool
        Add residuals.
    loss: Callable
        Optional Custom loss function. If None, the loss is CrossEntropy for classification tasks, or RMSE for regression.
    """

    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        activation: str = "relu",
        activation_out: Optional[list[tuple[str, int]]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.9, 0.999),
        n_iter: int = 1000,
        batch_size: int = 500,
        n_iter_print: int = 100,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = False,
        early_stopping: bool = True,
        residual: bool = False,
        loss: Optional[Callable] = None,
    ) -> None:
        super(MLP, self).__init__()
        activation = ACTIVATION_FUNCTIONS[activation] if activation in ACTIVATION_FUNCTIONS else None

        if n_units_in < 0:
            raise ValueError("n_units_in must be >= 0")
        if n_units_out < 0:
            raise ValueError("n_units_out must be >= 0")

        if residual:
            block = ResidualLayer
        else:
            block = LinearLayer

        # network
        layers = []

        if n_layers_hidden > 0:
            layers.append(
                block(
                    n_units_in,
                    n_units_hidden,
                    batch_norm=batch_norm,
                    activation=activation,
                )
            )
            n_units_hidden += int(residual) * n_units_in

            # add required number of layers
            for i in range(n_layers_hidden - 1):
                layers.append(
                    block(
                        n_units_hidden,
                        n_units_hidden,
                        batch_norm=batch_norm,
                        activation=activation,
                        dropout=dropout,
                    )
                )
                n_units_hidden += int(residual) * n_units_hidden

            # add final layers
            layers.append(nn.Linear(n_units_hidden, n_units_out))
        else:
            layers = [nn.Linear(n_units_in, n_units_out)]

        if activation_out is not None:
            total_nonlin_len = 0
            activations = []
            for nonlin, nonlin_len in activation_out:
                total_nonlin_len += nonlin_len
                activations.append((ACTIVATION_FUNCTIONS[nonlin](), nonlin_len))

            if total_nonlin_len != n_units_out:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {n_units_out}, but got {activation_out} with length {total_nonlin_len}"
                )
            layers.append(MultiActivationHead(activations))

        self.model = nn.Sequential(*layers)

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.opt_betas,
        )

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.patience = patience
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        Xt = self._check_tensor(X)
        yt = self._check_tensor(y)

        self._train(Xt, yt)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError(f"Invalid task type for predict_proba {self.task_type}")

        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            return yt.cpu().numpy().squeeze()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy().squeeze(), -1).squeeze()
            else:
                return yt.cpu().numpy().squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return np.mean(y_pred == y)
        else:
            return np.mean(np.inner(y - y_pred, y - y_pred) / 2.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float())

    def _train_epoch(self, loader: DataLoader) -> float:
        train_loss = []

        for batch_ndx, sample in enumerate(loader):
            self.optimizer.zero_grad()

            X_next, y_next = sample
            if len(X_next) < 2:
                continue

            preds = self.forward(X_next).squeeze()

            batch_loss = self.loss(preds, y_next)

            batch_loss.backward()

            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.optimizer.step()

            train_loss.append(batch_loss.detach())

        return torch.mean(torch.Tensor(train_loss))

    def _train(self, X: torch.Tensor, y: torch.Tensor) -> "MLP":
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().float()
        if self.task_type == "classification":
            y = y.long()

        # Load Dataset
        dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)

        # Setup the network and optimizer
        val_loss_best = 1e12
        patience = 0

        # do training
        for i in range(self.n_iter):
            self._train_epoch(loader)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset.tensors

                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X
        else:
            return torch.from_numpy(np.asarray(X))

    def __len__(self) -> int:
        return len(self.model)
