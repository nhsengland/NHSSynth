# Critical bug fixes and improvements debugged and implemented using Claude Code
# - 2026-01-16: Fixed kurtosis detection for adaptive temperature (lines 253-260)
# - 2026-01-16: Added comprehensive training monitoring (lines 506-569)
# - 2026-01-16: Added plot_training_curves() method for convergence diagnostics (lines 621-711)
# - 2026-01-19: Fixed posterior collapse with free bits (lines 391-405)
# - 2026-01-19: Added KL annealing and free bits parameters to train() (lines 466-510)
# - 2026-01-19: Added logsigma_z clamping for numerical stability (line 392)

# Set to True for verbose debug output during generation/training
DEBUG_VERBOSE = False

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
            "WeightedKLD",
            "Beta",
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

        N = N or self.nrows

        # Sample from learned latent distribution
        # NOTE: VAEs are designed to have N(0,1) latent space via KL divergence
        # The learned stats should be close to 0 and 1 - this is correct behavior!
        if hasattr(self, 'latent_mean') and hasattr(self, 'latent_std'):
            latent_mean = self.latent_mean.to(self.device)
            latent_std = self.latent_std.to(self.device)
            # Sample: z ~ N(learned_mean, learned_std)
            z_samples = latent_mean + latent_std * torch.randn(N, self.encoder.latent_dim, device=self.device)
        else:
            # Fallback to standard normal
            z_samples = torch.randn(N, self.encoder.latent_dim, device=self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a non-full backward hook")
            x_gen = self.decoder(z_samples)

        # Adaptive temperature scaling based on variable characteristics
        cont_idx = getattr(self.metatransformer, "continuous_value_indices", None)
        if cont_idx:
            # Adaptive temperature scaling based on variable characteristics
            # This prevents mode collapse while preserving naturally-peaked distributions
            base_temperature = 3.0  # Default for smooth distributions
            peaked_temperature = 1.5  # Lower for peaked distributions (high kurtosis)
            datetime_boost = 5.0  # Additional boost for datetime

            # Categorize continuous columns by their characteristics
            datetime_indices = []
            peaked_indices = []
            normal_indices = []

            cols = self.columns
            for idx in cont_idx:
                col_name = cols[idx]
                if col_name.endswith('_normalised'):
                    base_name = col_name.replace('_normalised', '')

                    # Check column characteristics from metadata
                    for col_meta in self.metatransformer._metadata:
                        if col_meta.name == base_name:
                            # Check if datetime
                            if hasattr(col_meta.transformer, '__class__') and 'DatetimeTransformer' in str(type(col_meta.transformer)):
                                datetime_indices.append(idx)
                            # Check if high kurtosis (peaked) - check outer transformer first
                            elif hasattr(col_meta.transformer, '_kurtosis') and col_meta.transformer._kurtosis > 5:
                                peaked_indices.append(idx)
                            else:
                                normal_indices.append(idx)
                            break

            # Apply adaptive temperature
            if peaked_indices:
                peaked_tensor = torch.tensor(peaked_indices, device=x_gen.device, dtype=torch.long)
                x_gen[:, peaked_tensor] = x_gen[:, peaked_tensor] * peaked_temperature

            if normal_indices:
                normal_tensor = torch.tensor(normal_indices, device=x_gen.device, dtype=torch.long)
                x_gen[:, normal_tensor] = x_gen[:, normal_tensor] * base_temperature

            if datetime_indices:
                datetime_tensor = torch.tensor(datetime_indices, device=x_gen.device, dtype=torch.long)
                x_gen[:, datetime_tensor] = x_gen[:, datetime_tensor] * (base_temperature * datetime_boost)

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

        from torch.distributions import OneHotCategorical
        for cat_idxs in categorical_groups:
            logits = x_gen[:, cat_idxs]
            x_gen_[:, cat_idxs] = OneHotCategorical(logits=logits).sample()

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

        # --- Z-jitter disabled ---
        # Adding jitter in z-space destroys the GMM structure that the decoder learned
        # The low decoder variance might be intentional: decoder outputs z≈0, relying on:
        #   1. Component selection for multimodality
        #   2. Component stds for variance during inverse transform
        # Mean shifts suggest component selection/means mismatch, not variance issue

        if torch.cuda.is_available():
            x_gen_ = x_gen_.cpu()

        arr = x_gen_.detach().cpu().numpy()
        df = pd.DataFrame(arr, columns=list(self.columns))
        return self.metatransformer.inverse_apply(df)


    def loss(self, X):
        mu_z, logsigma_z = self.encoder(X)

        # Clamp logsigma_z for numerical stability
        logsigma_z = torch.clamp(logsigma_z, min=-10, max=2)

        p = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q = Normal(mu_z, torch.exp(logsigma_z))

        # Compute per-dimension KLD for free bits
        kld_per_dim = torch.distributions.kl_divergence(q, p)  # Shape: (batch_size, latent_dim)

        # Apply free bits: only penalize KLD above threshold per dimension
        free_bits = getattr(self, '_free_bits', 0.0)  # Default: 0.0 (no free bits)
        if free_bits > 0:
            kld_per_dim = torch.maximum(kld_per_dim, torch.tensor(free_bits, device=X.device))

        kld = torch.sum(kld_per_dim)

        s = torch.randn_like(mu_z)
        z_samples = mu_z + s * torch.exp(logsigma_z)

        x_recon = self.decoder(z_samples)

        # Apply KL annealing via beta parameter
        beta = getattr(self, '_beta', 1.0)  # Default to 1.0 if not set

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

        # Apply beta weighting to KLD for annealing
        weighted_kld = beta * kld
        elbo = weighted_kld + reconstruction_loss

        return {
            "ELBO": elbo / X.size()[0],
            "ReconstructionLoss": reconstruction_loss / X.size()[0],
            "KLD": kld / X.size()[0],  # Unweighted KLD for monitoring
            "WeightedKLD": weighted_kld / X.size()[0],  # Weighted KLD used in loss
            "Beta": torch.tensor(beta),  # Current beta value
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
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_anneal_epochs: int = None,
        free_bits: float = 2.0,
    ) -> tuple[int, dict[str, list[float]]]:
        """
        Train the model with KL annealing and free bits to prevent posterior collapse.

        Args:
            num_epochs: Number of epochs to train for.
            patience: Number of epochs to wait for improvement before early stopping.
            displayed_metrics: List of metrics to display during training.
            notebook_run: Whether running in a notebook (affects progress bar display).
            beta_start: Starting value for KL weight (default: 0.0 for full annealing).
            beta_end: Final value for KL weight (default: 1.0 for standard VAE).
            beta_anneal_epochs: Number of epochs to anneal over (default: 50% of num_epochs).
                               If None, uses num_epochs // 2.
            free_bits: Minimum KLD per latent dimension (default: 2.0). Forces encoder to
                      use latent capacity by only penalizing KLD above this threshold.
                      Set to 0.0 to disable.

        Returns:
            The number of epochs trained for and a dictionary of the tracked metrics.
        """
        # Set default annealing period to 50% of training
        if beta_anneal_epochs is None:
            beta_anneal_epochs = num_epochs // 2

        tqdm.write(f"\nKL Annealing Schedule:")
        tqdm.write(f"  Beta: {beta_start:.4f} → {beta_end:.4f} over {beta_anneal_epochs} epochs")
        tqdm.write(f"  This prevents posterior collapse by gradually increasing KL weight")

        tqdm.write(f"\nFree Bits: {free_bits:.2f} per latent dimension")
        if free_bits > 0:
            tqdm.write(f"  Forces encoder to use latent capacity by not penalizing KLD below threshold")
        tqdm.write("")

        # Set free bits for loss function
        self._free_bits = free_bits

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
            # Update beta for KL annealing
            if epoch < beta_anneal_epochs:
                # Linear annealing from beta_start to beta_end
                self._beta = beta_start + (beta_end - beta_start) * (epoch / beta_anneal_epochs)
            else:
                # Hold at beta_end after annealing period
                self._beta = beta_end

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

            elbo_vals = self.metrics["ELBO"][-len(self.data_loader):]
            elbo = np.mean(elbo_vals) if len(elbo_vals) > 0 else float('nan')

            # Enhanced monitoring: display loss components every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                recon_vals = self.metrics["ReconstructionLoss"][-len(self.data_loader):]
                kld_vals = self.metrics["KLD"][-len(self.data_loader):]
                weighted_kld_vals = self.metrics["WeightedKLD"][-len(self.data_loader):]

                recon_loss = np.mean(recon_vals) if len(recon_vals) > 0 else float('nan')
                kld = np.mean(kld_vals) if len(kld_vals) > 0 else float('nan')
                weighted_kld = np.mean(weighted_kld_vals) if len(weighted_kld_vals) > 0 else float('nan')
                beta = self._beta  # Use current beta directly instead of averaging recorded values
                kld_ratio = kld / (recon_loss + 1e-8)

                tqdm.write(f"Epoch {epoch:3d}: ELBO={elbo:8.2f}, Recon={recon_loss:8.2f}, "
                          f"KLD={kld:8.2f} (β={beta:.3f}, weighted={weighted_kld:8.2f}), "
                          f"KLD/Recon={kld_ratio:.4f}")

                # Warning for posterior collapse (only after beta reaches 1.0)
                if beta > 0.9:  # Only warn when close to full KL weight
                    if kld < 10.0:
                        tqdm.write(f"  ⚠️  WARNING: Low KLD ({kld:.2f}) suggests posterior collapse!")
                    if kld_ratio < 0.01:
                        tqdm.write(f"  ⚠️  WARNING: KLD/Recon ratio very low ({kld_ratio:.4f}) - decoder ignoring latent!")

            if self._check_patience(epoch, elbo):
                num_epochs = epoch + 1
                break

        # Store learned latent statistics for generation
        if latent_mus:
            all_mus = torch.cat(latent_mus, dim=0)
            all_sigmas = torch.cat(latent_sigmas, dim=0)
            self.latent_mean = torch.mean(all_mus, dim=0)
            self.latent_std = torch.mean(all_sigmas, dim=0)

            # Enhanced latent statistics
            latent_mean_val = self.latent_mean.mean().item()
            latent_std_val = self.latent_std.mean().item()
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"TRAINING SUMMARY")
            tqdm.write(f"{'='*80}")
            tqdm.write(f"Learned latent stats: mean={latent_mean_val:.4f}, std={latent_std_val:.4f}")

            # Final metrics
            final_recon = np.mean(self.metrics["ReconstructionLoss"][-len(self.data_loader) :])
            final_kld = np.mean(self.metrics["KLD"][-len(self.data_loader) :])
            final_weighted_kld = np.mean(self.metrics["WeightedKLD"][-len(self.data_loader) :])
            final_beta = np.mean(self.metrics["Beta"][-len(self.data_loader) :])
            final_elbo = np.mean(self.metrics["ELBO"][-len(self.data_loader) :])

            tqdm.write(f"\nFinal Losses (per sample):")
            tqdm.write(f"  ELBO:              {final_elbo:8.2f}")
            tqdm.write(f"  Reconstruction:    {final_recon:8.2f}")
            tqdm.write(f"  KLD (unweighted):  {final_kld:8.2f}")
            tqdm.write(f"  KLD (weighted):    {final_weighted_kld:8.2f} (β={final_beta:.3f})")
            tqdm.write(f"  KLD/Recon ratio:   {final_kld / (final_recon + 1e-8):.4f}")

            # Convergence diagnostics
            tqdm.write(f"\nConvergence Diagnostics:")
            if final_kld < 10.0:
                tqdm.write(f"  ❌ POSTERIOR COLLAPSE: KLD={final_kld:.2f} is very low!")
                tqdm.write(f"     Decoder is ignoring the latent code and outputting near-constant values.")
                tqdm.write(f"     Solutions: Reduce KL weight (beta), use KL annealing, or train longer.")
            elif final_kld < 50.0:
                tqdm.write(f"  ⚠️  Mild posterior collapse: KLD={final_kld:.2f} is lower than ideal")
                tqdm.write(f"     Decoder may be under-utilizing latent information.")
            else:
                tqdm.write(f"  ✓ KLD={final_kld:.2f} appears healthy")

            if latent_std_val < 0.5:
                tqdm.write(f"  ⚠️  Low latent std ({latent_std_val:.4f}) - encoder is collapsing to deterministic")
            elif latent_std_val > 2.0:
                tqdm.write(f"  ⚠️  High latent std ({latent_std_val:.4f}) - encoder is too uncertain")
            else:
                tqdm.write(f"  ✓ Latent std={latent_std_val:.4f} is reasonable")

            tqdm.write(f"{'='*80}\n")
        else:
            # Fallback to standard normal
            self.latent_mean = torch.zeros(self.encoder.latent_dim)
            self.latent_std = torch.ones(self.encoder.latent_dim)

        self._finish_training(num_epochs)
        return (num_epochs, self.metrics)

    def plot_training_curves(self, save_path=None):
        """
        Plot training curves for ELBO, Reconstruction Loss, KLD, and Beta annealing.
        Useful for diagnosing convergence and posterior collapse.

        Args:
            save_path: Optional path to save the plot. If None, displays interactively.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()  # Flatten for easier indexing

        # Compute batch indices for x-axis
        batches = np.arange(len(self.metrics["ELBO"]))
        window = len(self.data_loader)

        # Plot 1: ELBO
        elbo_smooth = np.convolve(self.metrics["ELBO"], np.ones(window)/window, mode='valid')
        axes[0].plot(batches, self.metrics["ELBO"], alpha=0.3, label="ELBO (batch)")
        axes[0].plot(batches[window-1:], elbo_smooth, linewidth=2, label="ELBO (epoch avg)")
        axes[0].set_xlabel("Batch")
        axes[0].set_ylabel("ELBO (per sample)")
        axes[0].set_title("Evidence Lower Bound (ELBO)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Reconstruction Loss
        recon_smooth = np.convolve(self.metrics["ReconstructionLoss"], np.ones(window)/window, mode='valid')
        axes[1].plot(batches, self.metrics["ReconstructionLoss"], alpha=0.3, label="Recon (batch)")
        axes[1].plot(batches[window-1:], recon_smooth, linewidth=2, label="Recon (epoch avg)")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("Reconstruction Loss")
        axes[1].set_title("Reconstruction Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: KL Divergence (Unweighted)
        kld_smooth = np.convolve(self.metrics["KLD"], np.ones(window)/window, mode='valid')
        axes[2].plot(batches, self.metrics["KLD"], alpha=0.3, label="KLD (batch)")
        axes[2].plot(batches[window-1:], kld_smooth, linewidth=2, label="KLD (epoch avg)")
        axes[2].axhline(y=10, color='r', linestyle='--', alpha=0.5, label="Collapse threshold")
        axes[2].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label="Healthy threshold")
        axes[2].set_xlabel("Batch")
        axes[2].set_ylabel("KL Divergence (Unweighted)")
        axes[2].set_title("KL Divergence (Posterior Collapse if < 10)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Beta Annealing Schedule
        if "Beta" in self.metrics and len(self.metrics["Beta"]) > 0:
            beta_batches = np.arange(len(self.metrics["Beta"]))
            axes[3].plot(beta_batches, self.metrics["Beta"], linewidth=2, label="β")
            axes[3].set_xlabel("Batch")
            axes[3].set_ylabel("Beta (KL Weight)")
            axes[3].set_title("KL Annealing Schedule")
            axes[3].set_ylim(-0.1, 1.1)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, "No Beta data\n(KL annealing not used)",
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title("KL Annealing Schedule")

        # Plot 5: KLD/Recon Ratio
        kld_ratio = np.array(self.metrics["KLD"]) / (np.array(self.metrics["ReconstructionLoss"]) + 1e-8)
        ratio_smooth = np.convolve(kld_ratio, np.ones(window)/window, mode='valid')
        axes[4].plot(batches, kld_ratio, alpha=0.3, label="KLD/Recon (batch)")
        axes[4].plot(batches[window-1:], ratio_smooth, linewidth=2, label="KLD/Recon (epoch avg)")
        axes[4].axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label="Warning threshold")
        axes[4].set_xlabel("Batch")
        axes[4].set_ylabel("KLD / Reconstruction Ratio")
        axes[4].set_title("KLD/Recon Ratio (Low = Decoder Ignoring Latent)")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        # Plot 6: Weighted KLD (used in loss)
        if "WeightedKLD" in self.metrics and len(self.metrics["WeightedKLD"]) > 0:
            weighted_batches = np.arange(len(self.metrics["WeightedKLD"]))
            weighted_kld_smooth = np.convolve(self.metrics["WeightedKLD"], np.ones(window)/window, mode='valid')
            axes[5].plot(weighted_batches, self.metrics["WeightedKLD"], alpha=0.3, label="Weighted KLD (batch)")
            axes[5].plot(weighted_batches[window-1:], weighted_kld_smooth, linewidth=2, label="Weighted KLD (epoch avg)")
            axes[5].set_xlabel("Batch")
            axes[5].set_ylabel("Weighted KL Divergence")
            axes[5].set_title("Weighted KLD (β × KLD, used in loss)")
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, "No Weighted KLD data",
                        ha='center', va='center', transform=axes[5].transAxes)
            axes[5].set_title("Weighted KLD")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            tqdm.write(f"Training curves saved to {save_path}")
        else:
            plt.show()

        return fig
