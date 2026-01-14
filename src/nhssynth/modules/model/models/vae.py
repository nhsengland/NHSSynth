import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm

from nhssynth.common.constants import ACTIVATION_FUNCTIONS
from nhssynth.modules.model.common.model import Model


class Encoder(nn.Module):
    """Encoder, takes in x and outputs mu_z, sigma_z (diagonal Gaussian variational posterior assumed)"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        activation: str,
        learning_rate: float,
        shared_optimizer: bool,
    ) -> None:
        super().__init__()
        activation = ACTIVATION_FUNCTIONS[activation]
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        if not shared_optimizer:
            self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

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
        latent_dim: int,
        hidden_dim: int,
        activation: str,
        learning_rate: float,
        shared_optimizer: bool,
    ) -> None:
        super().__init__()
        activation = ACTIVATION_FUNCTIONS[activation]
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )
        if not shared_optimizer:
            self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, z):
        # No output activation - let the network learn the natural output distribution
        # Safety clipping and constraint repair will handle extreme values during generation
        return self.net(z)

        
class Noiser(nn.Module):
    def __init__(self, num_single_column_indices: list[int]) -> None:
        super().__init__()
        self.output_logsigma_fn = nn.Linear(num_single_column_indices, num_single_column_indices, bias=True)
        torch.nn.init.constant_(self.output_logsigma_fn.weight, 0.0)
        torch.nn.init.constant_(self.output_logsigma_fn.bias, -0.5)  # Start with scale ~0.6
        # Removed frozen line - now trainable!

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
        decoder_hidden_dim: int = 256,
        decoder_activation: str = "leaky_relu",
        decoder_learning_rate: float = 1e-3,
        shared_optimizer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.shared_optimizer = shared_optimizer
        self.encoder = Encoder(
            input_dim=self.ncols,
            latent_dim=encoder_latent_dim,
            hidden_dim=encoder_hidden_dim,
            activation=encoder_activation,
            learning_rate=encoder_learning_rate,
            shared_optimizer=self.shared_optimizer,
        ).to(self.device)
        self.decoder = Decoder(
            output_dim=self.ncols,
            latent_dim=decoder_latent_dim,
            hidden_dim=decoder_hidden_dim,
            activation=decoder_activation,
            learning_rate=decoder_learning_rate,
            shared_optimizer=self.shared_optimizer,
        ).to(self.device)
        num_continuous = len([idx for idx in self.single_column_indices
                     if idx in self.metatransformer.continuous_value_indices])
        self.noiser = Noiser(num_continuous).to(self.device)
        if self.shared_optimizer:
            assert (
                encoder_learning_rate == decoder_learning_rate
            ), "If `shared_optimizer` is True, `encoder_learning_rate` must equal `decoder_learning_rate`"
            self.optim = torch.optim.Adam(
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.noiser.parameters()),  # Added!
                lr=encoder_learning_rate,
            )
            self.zero_grad = self.optim.zero_grad
            self.step = self.optim.step
        else:
            self.zero_grad = lambda: (
                self.encoder.optim.zero_grad(),
                self.decoder.optim.zero_grad(),
            )
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

    @classmethod
    def get_metrics(cls) -> list[str]:
        return [
            "ELBO",
            "KLD",
            "ReconstructionLoss",
            "CategoricalLoss",
            "NumericalLoss",
            "BinaryLoss",
        ]

    def reconstruct(self, X):
        mu_z, logsigma_z = self.encoder(X)
        x_recon = self.decoder(mu_z)
        return x_recon

    def generate(self, N: Optional[int] = None) -> pd.DataFrame:
        import re
        import torch
        from tqdm import tqdm
        import numpy as np

        N = N or self.nrows

        # Sample from learned latent distribution
        # NOTE: VAEs are designed to have N(0,1) latent space via KL divergence
        # The learned stats should be close to 0 and 1 - this is correct behavior!
        if hasattr(self, 'latent_mean') and hasattr(self, 'latent_std'):
            latent_mean = self.latent_mean.to(self.device)
            latent_std = self.latent_std.to(self.device)
            # Sample: z ~ N(learned_mean, learned_std)
            z_samples = latent_mean + latent_std * torch.randn(N, self.encoder.latent_dim, device=self.device)
            tqdm.write(f"Sampling from learned posterior: mean={latent_mean.mean():.3f}, std={latent_std.mean():.3f}")
            tqdm.write(f"  (Note: VAE latent space is regularized toward N(0,1) by design)")
        else:
            # Fallback to standard normal
            z_samples = torch.randn(N, self.encoder.latent_dim, device=self.device)
            tqdm.write("Using N(0,1) prior for latent sampling")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a non-full backward hook")
            x_gen = self.decoder(z_samples)

        # Diagnostic: Check decoder output statistics before any modifications
        cont_idx = getattr(self.metatransformer, "continuous_value_indices", None)
        if cont_idx:
            cont_idx_tensor = torch.tensor(cont_idx, device=x_gen.device, dtype=torch.long)
            decoder_mean = float(x_gen[:, cont_idx_tensor].mean())
            decoder_std = float(x_gen[:, cont_idx_tensor].std())
            decoder_min = float(x_gen[:, cont_idx_tensor].min())
            decoder_max = float(x_gen[:, cont_idx_tensor].max())
            tqdm.write(f"Decoder outputs for {len(cont_idx)} continuous value columns: "
                      f"mean={decoder_mean:.3f}, std={decoder_std:.3f}, "
                      f"range=[{decoder_min:.3f}, {decoder_max:.3f}]")

            # Temperature scaling to increase variance of generated continuous values
            # This prevents mode collapse onto GMM component means
            temperature = 3.0  # Aggressive scaling to smooth out GMM component peaks

            # Identify datetime columns (they have "_normalised" suffix and original column is datetime)
            datetime_indices = []
            cols = self.columns
            for idx in cont_idx:
                col_name = cols[idx]
                # Datetime columns have format "dob_normalised" - check if original column was datetime
                if col_name.endswith('_normalised'):
                    base_name = col_name.replace('_normalised', '')
                    # Check if this is a datetime column by looking for it in metadata
                    for col_meta in self.metatransformer._metadata:
                        if col_meta.name == base_name and hasattr(col_meta.transformer, '__class__'):
                            if 'DatetimeTransformer' in str(type(col_meta.transformer)):
                                datetime_indices.append(idx)
                                break

            # Apply base temperature to all continuous columns
            x_gen[:, cont_idx_tensor] = x_gen[:, cont_idx_tensor] * temperature

            # Apply additional temperature boost to datetime columns (single Gaussian needs more spread)
            if datetime_indices:
                datetime_boost = 5.0  # Additional 5x for datetime (total 15x for wide temporal distributions)
                datetime_tensor = torch.tensor(datetime_indices, device=x_gen.device, dtype=torch.long)
                x_gen[:, datetime_tensor] = x_gen[:, datetime_tensor] * datetime_boost
                tqdm.write(f"Applied temperature scaling: {temperature}x to continuous, {temperature * datetime_boost}x to {len(datetime_indices)} datetime columns")
            else:
                tqdm.write(f"Applied temperature scaling: {temperature}x to continuous columns")

            # REMOVED aggressive safety clipping - let transformer handle it

        # after x_gen = self.decoder(...)
        x_gen_ = x_gen.clone()

        import re
        cols = self.columns  # make sure MetaTransformer.transform() set this!
        categorical_groups = []
        gmm_component_groups = []
        for group in (self.multi_column_indices or []):
            names = [cols[j] for j in group]
            has_value = any(n.endswith(("_value","_normalized","_normalised")) for n in names)
            has_mix  = any(re.search(r"_c\d+$", n) for n in names)
            if has_mix:
                # GMM component columns - apply temperature to smooth component selection
                gmm_component_groups.append(group)
            elif not has_value:
                # Regular categorical variables
                categorical_groups.append(group)

        # Apply temperature to GMM component logits to encourage mixing
        gmm_temperature = 2.0  # Moderate temperature to preserve multimodal structure
        for gmm_idxs in gmm_component_groups:
            x_gen_[:, gmm_idxs] = x_gen[:, gmm_idxs] / gmm_temperature
        if gmm_component_groups:
            tqdm.write(f"Applied GMM component temperature {gmm_temperature}x to {len(gmm_component_groups)} groups")

        from torch.distributions import OneHotCategorical
        for cat_idxs in categorical_groups:
            logits = x_gen[:, cat_idxs]
            x_gen_[:, cat_idxs] = OneHotCategorical(logits=logits).sample()

        tqdm.write(f"one-hot groups (sizes): {[len(g) for g in categorical_groups]}")

        # --- singles noise (existing behaviour) ---
        cont_idx = getattr(self.metatransformer, "continuous_value_indices", None)
        if cont_idx:
            cont_idx_list = [idx for idx in self.single_column_indices if idx in cont_idx]
            if cont_idx_list:
                idx_tensor = torch.tensor(cont_idx_list, device=x_gen.device, dtype=torch.long)
                loc = x_gen[:, cont_idx_list]
                noiser_output = self.noiser(loc)
                scale = torch.exp(noiser_output)
                x_gen_[:, idx_tensor] = loc + scale * torch.randn_like(loc)

        # --- DISABLED z-jitter for debugging ---
        # The decoder outputs already have high std (>2), adding jitter makes it worse
        # TODO: Re-enable with proper calibration once decoder outputs are stable
        cont_idx = getattr(self.metatransformer, "continuous_value_indices", None)
        if cont_idx and False:  # Disabled
            idx = torch.tensor(cont_idx, device=x_gen.device, dtype=torch.long)
            z_sigma = getattr(self, "z_jitter_std", 0.0)  # Set to 0 by default
            if not torch.is_tensor(z_sigma):
                z_sigma = torch.tensor(float(z_sigma), device=x_gen.device)
            if z_sigma > 0:
                x_gen_[:, idx] = x_gen[:, idx] + z_sigma * torch.randn_like(x_gen[:, idx])
                tqdm.write(f"Applied z-jitter with sigma={float(z_sigma):.3f}")
            else:
                tqdm.write(f"Z-jitter disabled (sigma=0)")

        if torch.cuda.is_available():
            x_gen_ = x_gen_.cpu()

        df = pd.DataFrame(x_gen_.detach().numpy(), columns=self.columns)
        
        # Debug: check z dispersion
        try:
            zcols = [j for j,n in enumerate(self.columns) if n.endswith(("_value","_normalized","_normalised"))]
            if zcols:
                z = x_gen_[:, zcols].detach().cpu().numpy()
                tqdm.write(f"z std (median over cols): {np.median(np.std(z, axis=0))}")
        except Exception:
            pass

        # --- per-feature z stds for debug ---
        def _zcol_for(base: str):
            suffixes = ("_value", "_normalized", "_normalised")
            # make a plain list to be safe with both Index and list
            cols = list(self.columns)
            for sfx in suffixes:
                name = f"{base}{sfx}"
                if name in cols:
                    return cols.index(name)  # position in the transformed matrix
            return None

        for base in ("x8", "dob"):
            j = _zcol_for(base)
            if j is not None:
                zvals = x_gen_[:, j].detach().cpu().numpy()
                tqdm.write(f"[gen:z-std] {base}: std={float(np.std(zvals)):.4f}")
            else:
                tqdm.write(f"[gen:z-std] {base}: NO Z-COLUMN FOUND")
        
        
        # BEFORE calling inverse_apply
        arr = x_gen_.detach().cpu().numpy()   # <-- ensure a real NumPy array
        df  = pd.DataFrame(arr, columns=list(self.columns))  # columns as plain list is safest
        return self.metatransformer.inverse_apply(df)


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

        gauss_loglik = torch.tensor(0.0, device=X.device)
        binary_loglik = torch.tensor(0.0, device=X.device)

        if self.single_column_indices:
            # Separate continuous (z-score) columns from binary (missingness) columns
            cont_indices = [idx for idx in self.single_column_indices
                        if idx in self.metatransformer.continuous_value_indices]
            miss_indices = [idx for idx in self.single_column_indices
                        if idx not in self.metatransformer.continuous_value_indices]

            # Gaussian loss for continuous columns
            if cont_indices:
                loc = x_recon[:, cont_indices]
                noiser_output = self.noiser(x_recon[:, cont_indices])
                scale = torch.exp(noiser_output)
                gauss_loglik = Normal(loc=loc, scale=scale).log_prob(X[:, cont_indices]).sum()

            # BCE loss for binary missingness columns
            if miss_indices:
                logits = x_recon[:, miss_indices]
                targets = X[:, miss_indices]
                binary_loglik = -torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction='sum'
                )

        reconstruction_loss = -(categoric_loglik + gauss_loglik + binary_loglik)

        elbo = kld + reconstruction_loss

        return {
            "ELBO": elbo / X.size()[0],
            "ReconstructionLoss": reconstruction_loss / X.size()[0],
            "KLD": kld / X.size()[0],
            "CategoricalLoss": categoric_loglik / X.size()[0],
            "NumericalLoss": gauss_loglik / X.size()[0],
            "BinaryLoss": binary_loglik / X.size()[0],
        }

    def train(
        self,
        num_epochs: int = 100,
        patience: int = 5,
        displayed_metrics: list[str] = ["ELBO"],
        notebook_run: bool = False,
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
        self._start_training(num_epochs, patience, displayed_metrics, notebook_run)

        self.encoder.train()
        self.decoder.train()
        self.noiser.train()

        # Initialize latent statistics accumulators
        latent_mus = []
        latent_sigmas = []

        for epoch in tqdm(
            range(num_epochs),
            desc="Epochs",
            position=len(self.stats_bars) if not notebook_run else 0,
            leave=False,
        ):
            if not notebook_run:
                epoch_progress = tqdm(
                    self.data_loader,
                    desc="Batches",
                    position=len(self.stats_bars) + 1,
                    leave=False,
                )
            else:
                epoch_progress = self.data_loader
            for (Y_subset,) in epoch_progress:
                self.zero_grad()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Using a non-full backward hook")
                    losses = self.loss(Y_subset.to(self.device))

                    # Track latent statistics from last few epochs for generation
                    if epoch >= max(0, num_epochs - 5):
                        with torch.no_grad():
                            mu_z, logsigma_z = self.encoder(Y_subset.to(self.device))
                            latent_mus.append(mu_z.cpu())
                            latent_sigmas.append(torch.exp(logsigma_z).cpu())

                losses["ELBO"].backward()
                self.step()
                self._record_metrics(losses)

            elbo = np.mean(self.metrics["ELBO"][-len(self.data_loader) :])
            if self._check_patience(epoch, elbo):
                num_epochs = epoch + 1
                break

        # Store learned latent statistics for generation
        if latent_mus:
            all_mus = torch.cat(latent_mus, dim=0)
            all_sigmas = torch.cat(latent_sigmas, dim=0)
            self.latent_mean = torch.mean(all_mus, dim=0)
            self.latent_std = torch.mean(all_sigmas, dim=0)
            tqdm.write(f"Learned latent stats: mean={self.latent_mean.mean():.3f}, std={self.latent_std.mean():.3f}")
        else:
            # Fallback to standard normal
            self.latent_mean = torch.zeros(self.encoder.latent_dim)
            self.latent_std = torch.ones(self.encoder.latent_dim)

        self._finish_training(num_epochs)
        return (num_epochs, self.metrics)
