from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nhssynth.modules.model.common.ctgan_sampler import CTGANConditionalSampler, extract_categorical_groups
from nhssynth.modules.model.common.mlp import MLP
from nhssynth.modules.model.models.gan import GAN


class CTGAN(GAN):
    """
    Conditional Tabular GAN (Xu et al. 2019) adapted for NHSSynth.

    Extends the base WGAN-GP GAN with three key additions:

    1. **Conditional vector** — at each training step a categorical column is
       sampled uniformly and a category is drawn from its empirical distribution.
       A one-hot condition vector spanning all categorical columns is appended to
       the generator's noise input and to the discriminator's data input.

    2. **Conditional training sampler** — real training rows are resampled so that
       the conditioned-on category is always active, ensuring balanced coverage of
       all categorical modes during training.

    3. **PacGAN discriminator** — processes ``pac`` (default 2) samples concatenated
       along the feature dimension, preventing mode collapse by giving the
       discriminator a broader view of the generated distribution.

    4. **Conditional cross-entropy loss** — the generator is penalised for failing
       to reproduce the conditioned-on category in its output.

    Args:
        pac: Number of samples packed together for the PacGAN discriminator (default 2).
        lambda_cond: Weight of the conditional cross-entropy loss (default 1.0).
        All other args are inherited from :class:`GAN`.
    """

    def __init__(
        self,
        *args,
        pac: int = 2,
        lambda_cond: float = 1.0,
        **kwargs,
    ) -> None:
        # Initialise the base GAN first (builds generator/discriminator with wrong dims —
        # we rebuild them below once we know cond_dim).
        super(CTGAN, self).__init__(*args, **kwargs)

        self.pac = pac
        self.lambda_cond = lambda_cond

        # Identify categorical column groups from the metatransformer metadata
        self.categorical_groups = extract_categorical_groups(
            self.multi_column_indices, self.columns
        )

        # Store the full training tensor for the conditional sampler
        # (dataset is a TensorDataset; first element is the data tensor)
        self._tensor_data: torch.Tensor = self.data_loader.dataset.tensors[0]

        if len(self.categorical_groups) > 0:
            self.ctgan_sampler = CTGANConditionalSampler(
                self._tensor_data, self.categorical_groups
            )
            self.cond_dim: int = self.ctgan_sampler.cond_dim
        else:
            # Purely continuous dataset — no conditional sampling, behave like plain GAN
            self.ctgan_sampler = None
            self.cond_dim = 0

        # Rebuild generator and discriminator with CTGAN-correct input dimensions.
        # (The base GAN __init__ already stored the hyperparams we need.)
        gen_kwargs = dict(
            n_units_in=self.noise_dim + self.cond_dim,
            n_units_out=self.ncols,
            n_layers_hidden=kwargs.get("generator_n_layers_hidden", 2),
            n_units_hidden=kwargs.get("generator_n_units_hidden", 250),
            activation=kwargs.get("generator_activation", "leaky_relu"),
            batch_norm=kwargs.get("generator_batch_norm", False),
            dropout=kwargs.get("generator_dropout", 0.0),
            lr=kwargs.get("generator_lr", 2e-4),
            residual=kwargs.get("generator_residual", True),
            opt_betas=kwargs.get("generator_opt_betas", (0.9, 0.999)),
        )
        disc_kwargs = dict(
            n_units_in=pac * (self.ncols + self.cond_dim),
            n_units_out=1,
            n_layers_hidden=kwargs.get("discriminator_n_layers_hidden", 3),
            n_units_hidden=kwargs.get("discriminator_n_units_hidden", 300),
            activation=kwargs.get("discriminator_activation", "leaky_relu"),
            activation_out=None,
            batch_norm=kwargs.get("discriminator_batch_norm", False),
            dropout=kwargs.get("discriminator_dropout", 0.1),
            lr=kwargs.get("discriminator_lr", 2e-4),
            opt_betas=kwargs.get("discriminator_opt_betas", (0.9, 0.999)),
        )
        self.generator = MLP(**gen_kwargs).to(self.device)
        self.discriminator = MLP(**disc_kwargs).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_args(cls) -> list[str]:
        return GAN.get_args() + ["pac", "lambda_cond"]

    def generate(self, N: int, cond: Optional[np.ndarray] = None) -> pd.DataFrame:
        N = N or self.nrows
        self.generator.eval()
        with torch.no_grad():
            fake = self(N).detach().cpu().numpy()
        return self.metatransformer.inverse_apply(
            pd.DataFrame(fake, columns=self.columns)
        )

    def forward(self, N: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn(N, self.noise_dim, device=self.device)
        if self.ctgan_sampler is not None:
            condvec, _, _, _ = self.ctgan_sampler.sample_condvec(N)
            condvec = condvec.to(self.device)
            noise = torch.cat([noise, condvec], dim=1)
        return self.generator(noise)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Tuple[float, float]:
        """Override GAN's _train_epoch to use the CTGAN conditional sampler."""
        if self.ctgan_sampler is None:
            # No categorical columns — fall back to plain GAN behaviour
            return super()._train_epoch()

        g_losses, d_losses = [], []
        batch_size = self.batch_size

        # Iterate the same number of steps as the standard data loader would
        n_steps = max(1, self.nrows // batch_size)

        for _ in tqdm(
            range(n_steps),
            desc="Batches",
            position=len(self.stats_bars) + 1,
            leave=False,
        ):
            bs = min(batch_size, self.nrows)

            # --- Discriminator step ---
            condvec, mask, col_idxs, cat_idxs = self.ctgan_sampler.sample_condvec(bs)
            condvec = condvec.to(self.device)
            mask = mask.to(self.device)
            real = self.ctgan_sampler.sample_data_conditioned(bs, col_idxs, cat_idxs).to(self.device)

            d_loss = self._ctgan_discriminator_step(real, condvec)
            d_losses.append(d_loss)

            # --- Generator step ---
            g_loss = self._ctgan_generator_step(bs)
            g_losses.append(g_loss)

            self._record_metrics({"DLoss": d_loss, "GLoss": g_loss})

        return float(np.mean(g_losses)), float(np.mean(d_losses))

    def _ctgan_discriminator_step(
        self,
        real: torch.Tensor,
        condvec: torch.Tensor,
    ) -> float:
        self.discriminator.train()
        self.discriminator.optim.zero_grad()

        bs = real.shape[0]

        # For PacGAN, gather `pac` pairs of (real, fake) and pack them
        real_packed_parts = []
        fake_packed_parts = []

        for _ in range(self.pac):
            # Real side: resample with a fresh condvec
            cv, _, ci, ki = self.ctgan_sampler.sample_condvec(bs)
            cv = cv.to(self.device)
            r = self.ctgan_sampler.sample_data_conditioned(bs, ci, ki).to(self.device)
            real_packed_parts.append(torch.cat([r, cv], dim=1))

            # Fake side
            noise = torch.randn(bs, self.noise_dim, device=self.device)
            fake = self.generator(torch.cat([noise, cv], dim=1))
            fake_packed_parts.append(torch.cat([fake, cv], dim=1))

        real_packed = torch.cat(real_packed_parts, dim=1)  # (B, pac*(ncols+cond))
        fake_packed = torch.cat(fake_packed_parts, dim=1)

        # WGAN-GP losses
        errD_real = self.discriminator(real_packed).squeeze().mean()
        errD_fake = self.discriminator(fake_packed.detach()).squeeze().mean()

        # Gradient penalty on interpolated packed samples
        alpha = torch.rand(bs, 1, device=self.device)
        interp = (alpha * real_packed + (1 - alpha) * fake_packed).requires_grad_(True)
        d_interp = self.discriminator(interp).squeeze()
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gp = self.lambda_gradient_penalty * ((grads.norm(2, dim=1) - 1) ** 2).mean()

        errD = -errD_real + errD_fake + gp
        errD.backward()

        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clipping_value)
        self.discriminator.optim.step()

        if np.isnan(errD.item()):
            raise RuntimeError("NaNs detected in the CTGAN discriminator loss")
        return errD.item()

    def _ctgan_generator_step(self, batch_size: int) -> float:
        self.generator.train()
        self.generator.optim.zero_grad()

        fake_parts = []
        cond_parts = []
        mask_parts = []

        for _ in range(self.pac):
            cv, msk, _, _ = self.ctgan_sampler.sample_condvec(batch_size)
            cv = cv.to(self.device)
            msk = msk.to(self.device)
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake = self.generator(torch.cat([noise, cv], dim=1))
            fake_parts.append(torch.cat([fake, cv], dim=1))
            cond_parts.append(cv)
            mask_parts.append(msk)

        fake_packed = torch.cat(fake_parts, dim=1)

        errG_w = -self.discriminator(fake_packed).squeeze().mean()

        # Conditional cross-entropy loss (sum over pac copies)
        errG_cond = sum(
            self._conditional_loss(fp[:, : self.ncols], cv, msk)
            for fp, cv, msk in zip(fake_parts, cond_parts, mask_parts)
        )

        errG = errG_w + self.lambda_cond * errG_cond
        errG.backward()

        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clipping_value)
        self.generator.optim.step()

        if torch.isnan(errG):
            raise RuntimeError("NaNs detected in the CTGAN generator loss")
        return errG.item()

    def _conditional_loss(
        self,
        fake: torch.Tensor,
        condvec: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy between the generator's output and the conditioned-on category,
        weighted by the column mask so only the selected column contributes.
        """
        total = torch.tensor(0.0, device=self.device)
        if self.ctgan_sampler is None or len(self.categorical_groups) == 0:
            return total

        for col_idx, group in enumerate(self.categorical_groups):
            offset = self.ctgan_sampler._offsets[col_idx]
            n_cat = len(group)

            fake_logits = fake[:, group]  # (B, n_cat) — raw generator outputs
            target = condvec[:, offset : offset + n_cat]  # (B, n_cat) — one-hot target
            col_mask = mask[:, col_idx]  # (B,)

            log_prob = F.log_softmax(fake_logits, dim=1)
            ce = -(target * log_prob).sum(dim=1)  # (B,)
            total += (col_mask * ce).mean()

        return total
