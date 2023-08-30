import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nhssynth.common.constants import ACTIVATION_FUNCTIONS
from nhssynth.modules.model.common.model import Model
from torch.distributions.normal import Normal
from tqdm import tqdm


class Encoder(nn.Module):
    """Encoder, takes in x and outputs mu_z, sigma_z (diagonal Gaussian variational posterior assumed)"""

    def __init__(
        self,
        input_dim: int,
        encoder_latent_dim: int,
        encoder_hidden_dim: int,
        encoder_activation: str,
        encoder_learning_rate: float,
        shared_optimizer: bool,
    ) -> None:
        super().__init__()
        activation = ACTIVATION_FUNCTIONS[encoder_activation]
        self.latent_dim = encoder_latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            activation(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            activation(),
            nn.Linear(encoder_hidden_dim, 2 * encoder_latent_dim),
        )
        if not shared_optimizer:
            self.optim = torch.optim.Adam(self.parameters(), lr=encoder_learning_rate)

    def forward(self, x):
        outs = self.net(x)
        mu_z = outs[:, : self.latent_dim]
        logsigma_z = outs[:, self.latent_dim :]
        return mu_z, logsigma_z


class Decoder(nn.Module):
    """Decoder, takes in z and outputs reconstruction"""

    def __init__(
        self,
        output_dim: int,
        decoder_latent_dim: int,
        decoder_hidden_dim: int,
        decoder_activation: str,
        decoder_learning_rate: float,
        shared_optimizer: bool,
    ) -> None:
        super().__init__()
        activation = ACTIVATION_FUNCTIONS[decoder_activation]
        self.net = nn.Sequential(
            nn.Linear(decoder_latent_dim, decoder_hidden_dim),
            activation(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            activation(),
            nn.Linear(decoder_hidden_dim, output_dim),
        )
        if not shared_optimizer:
            self.optim = torch.optim.Adam(self.parameters(), lr=decoder_learning_rate)

    def forward(self, z):
        return self.net(z)


class Noiser(nn.Module):
    def __init__(
        self,
        num_single_column_indices: list[int],
    ) -> None:
        super().__init__()
        self.output_logsigma_fn = nn.Linear(num_single_column_indices, num_single_column_indices, bias=True)
        torch.nn.init.zeros_(self.output_logsigma_fn.weight)
        torch.nn.init.zeros_(self.output_logsigma_fn.bias)
        self.output_logsigma_fn.weight.requires_grad = False

    def forward(self, X):
        return self.output_logsigma_fn(X)


class VAE(Model):
    """
    A Variational Autoencoder (VAE) model. Accepts [`Model`][nhssynth.modules.model.common.model.Model] arguments as well as the following:

    Args:
        encoder_latent_dim: The dimensionality of the latent space.
        encoder_hidden_dim: The dimensionality of the hidden layers in the encoder.
        encoder_activation: The activation function to use in the encoder.
        encoder_learning_rate: The learning rate for the encoder.
        decoder_latent_dim: The dimensionality of the hidden layers in the decoder.
        decoder_hidden_dim: The dimensionality of the hidden layers in the decoder.
        decoder_activation: The activation function to use in the decoder.
        decoder_learning_rate: The learning rate for the decoder.
        shared_optimizer: Whether to use a shared optimizer for the encoder and decoder.
    """

    def __init__(
        self,
        *args,
        encoder_latent_dim: int = 256,
        encoder_hidden_dim: int = 256,
        encoder_activation: str = "leaky_relu",
        encoder_learning_rate: float = 1e-3,
        decoder_latent_dim: int = 256,
        decoder_hidden_dim: int = 32,
        decoder_activation: str = "leaky_relu",
        decoder_learning_rate: float = 1e-3,
        shared_optimizer: bool = True,
        **kwargs,
    ) -> None:
        super(VAE, self).__init__(*args, **kwargs)

        self.shared_optimizer = shared_optimizer
        self.encoder = Encoder(
            self.ncols,
            encoder_latent_dim,
            encoder_hidden_dim,
            encoder_activation,
            encoder_learning_rate,
            self.shared_optimizer,
        ).to(self.device)
        self.decoder = Decoder(
            self.ncols,
            decoder_latent_dim,
            decoder_hidden_dim,
            decoder_activation,
            decoder_learning_rate,
            self.shared_optimizer,
        ).to(self.device)
        self.noiser = Noiser(
            len(self.single_column_indices),
        ).to(self.device)
        if self.shared_optimizer:
            assert (
                encoder_learning_rate == decoder_learning_rate
            ), "If `shared_optimizer` is True, `encoder_learning_rate` must equal `decoder_learning_rate`"
            self.optim = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=encoder_learning_rate,
            )
            self.zero_grad = self.optim.zero_grad
            self.step = self.optim.step
        else:
            self.zero_grad = lambda: (self.encoder.optim.zero_grad(), self.decoder.optim.zero_grad())
            self.step = lambda: (self.encoder.optim.step(), self.decoder.optim.step())

    @classmethod
    def get_args(cls) -> list[str]:
        return [
            "encoder_latent_dim",
            "encoder_hidden_dim",
            "encoder_activation",
            "encoder_learning_rate",
            "decoder_latent_dim",
            "decoder_hidden_dim",
            "decoder_activation",
            "decoder_learning_rate",
            "shared_optimizer",
        ]

    def reconstruct(self, X):
        mu_z, logsigma_z = self.encoder(X)
        x_recon = self.decoder(mu_z)
        return x_recon

    def generate(self, N: Optional[int] = None) -> pd.DataFrame:
        N = N or self.nrows
        z_samples = torch.randn_like(torch.ones((N, self.encoder.latent_dim)), device=self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a non-full backward hook")
            x_gen = self.decoder(z_samples)
        x_gen_ = torch.ones_like(x_gen, device=self.device)

        if self.multi_column_indices != [[]]:
            for cat_idxs in self.multi_column_indices:
                x_gen_[:, cat_idxs] = torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits=x_gen[:, cat_idxs]
                ).sample()

        x_gen_[:, self.single_column_indices] = x_gen[:, self.single_column_indices] + torch.exp(
            self.noiser(x_gen[:, self.single_column_indices])
        ) * torch.randn_like(x_gen[:, self.single_column_indices])
        if torch.cuda.is_available():
            x_gen_ = x_gen_.cpu()
        return self.metatransformer.inverse_apply(pd.DataFrame(x_gen_.detach(), columns=self.columns))

    def loss(self, X):
        mu_z, logsigma_z = self.encoder(X)

        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q = Normal(mu_z, torch.exp(logsigma_z))

        kld = torch.sum(torch.distributions.kl_divergence(q, p))

        s = torch.randn_like(mu_z)
        z_samples = mu_z + s * torch.exp(logsigma_z)

        x_recon = self.decoder(z_samples)

        categoric_loglik = 0

        if self.multi_column_indices != [[]]:
            for cat_idxs in self.multi_column_indices:
                categoric_loglik += -torch.nn.functional.cross_entropy(
                    x_recon[:, cat_idxs],
                    torch.max(X[:, cat_idxs], 1)[1],
                ).sum()

        gauss_loglik = 0
        if self.single_column_indices:
            gauss_loglik = (
                Normal(
                    loc=x_recon[:, self.single_column_indices],
                    scale=torch.exp(self.noiser(x_recon[:, self.single_column_indices])),
                )
                .log_prob(X[:, self.single_column_indices])
                .sum()
            )

        reconstruction_loss = -(categoric_loglik + gauss_loglik)

        elbo = kld + reconstruction_loss

        return {
            "ELBO": elbo / X.size()[0],
            "ReconstructionLoss": reconstruction_loss / X.size()[0],
            "KLD": kld / X.size()[0],
            "CategoricalLoss": categoric_loglik / X.size()[0],
            "NumericalLoss": gauss_loglik / X.size()[0],
        }

    def train(
        self,
        num_epochs: int = 100,
        patience: int = 5,
        displayed_metrics: list[str] = ["ELBO"],
    ) -> tuple[int, dict[str, list[float]]]:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train for.
            patience: Number of epochs to wait for improvement before early stopping.
            displayed_metrics: List of metrics to display during training.

        Returns:
            The number of epochs trained for and a dictionary of the tracked metrics.
        """
        print("")
        self._start_training(num_epochs, patience, displayed_metrics)

        self.encoder.train()
        self.decoder.train()
        self.noiser.train()

        for epoch in tqdm(range(num_epochs), desc="Epochs", position=len(self.stats_bars), leave=False):
            for (Y_subset,) in tqdm(self.data_loader, desc="Batches", position=len(self.stats_bars) + 1, leave=False):
                self.zero_grad()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Using a non-full backward hook")
                    losses = self.loss(Y_subset.to(self.device))
                losses["ELBO"].backward()
                self.step()
                self._record_metrics(losses)

            elbo = np.mean(self.metrics["ELBO"][-len(self.data_loader) :])
            if self._check_patience(epoch, elbo):
                num_epochs = epoch + 1
                break

        self._finish_training(num_epochs)
        return (num_epochs, self.metrics)
