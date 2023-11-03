from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nhssynth.modules.model.common.dp import DPMixin
from nhssynth.modules.model.common.mlp import MLP
from nhssynth.modules.model.common.model import Model


class GAN(Model):
    """
    Basic GAN implementation.

    Args:
        n_units_conditional: int
            Number of conditional units
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_activation: string, default 'elu'
            Nonlinearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        generator_n_iter: int
            Maximum number of iterations in the Generator.
        generator_batch_norm: bool
            Enable/disable batch norm for the generator
        generator_dropout: float
            Dropout value. If 0, the dropout is not used.
        generator_residual: bool
            Use residuals for the generator
        generator_activation_out: Optional[List[Tuple[str, int]]]
            List of activations. Useful with the TabularEncoder
        generator_lr: float = 2e-4
            Generator learning rate, used by the Adam optimizer
        generator_weight_decay: float = 1e-3
            Generator weight decay, used by the Adam optimizer
        generator_opt_betas: tuple = (0.9, 0.999)
            Generator initial decay rates, used by the Adam Optimizer
        generator_extra_penalty_cbks: List[Callable]
            Additional loss callabacks for the generator. Used by the TabularGAN for the conditional loss
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_activation: string, default 'relu'
            Nonlinearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_batch_norm: bool
            Enable/disable batch norm for the discriminator
        discriminator_dropout: float
            Dropout value for the discriminator. If 0, the dropout is not used.
        discriminator_lr: float
            Discriminator learning rate, used by the Adam optimizer
        discriminator_weight_decay: float
            Discriminator weight decay, used by the Adam optimizer
        discriminator_opt_betas: tuple
            Initial weight decays for the Adam optimizer
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
    """

    def __init__(
        self,
        *args,
        n_units_conditional: int = 0,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_activation: str = "leaky_relu",
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_lr: float = 2e-4,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_activation: str = "leaky_relu",
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 2e-4,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        **kwargs,
    ) -> None:
        super(GAN, self).__init__(*args, **kwargs)

        self.generator_n_units_hidden = generator_n_units_hidden
        self.n_units_conditional = n_units_conditional

        self.generator = MLP(
            n_units_in=generator_n_units_hidden + n_units_conditional,
            n_units_out=self.ncols,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            activation=generator_activation,
            # nonlin_out=generator_activation_out,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            lr=generator_lr,
            residual=generator_residual,
            opt_betas=generator_opt_betas,
        ).to(self.device)

        self.discriminator = MLP(
            n_units_in=self.ncols + n_units_conditional,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            activation=discriminator_activation,
            activation_out=[("none", 1)],
            batch_norm=discriminator_batch_norm,
            dropout=discriminator_dropout,
            lr=discriminator_lr,
            opt_betas=discriminator_opt_betas,
        ).to(self.device)

        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty

        def gen_fake_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(X),), device=self.device)

        def gen_true_labels(X: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(X),), device=self.device)

        self.fake_labels_generator = gen_fake_labels
        self.true_labels_generator = gen_true_labels

    @classmethod
    def get_args(cls) -> list[str]:
        return [
            "n_units_conditional",
            "generator_n_layers_hidden",
            "generator_n_units_hidden",
            "generator_activation",
            "generator_batch_norm",
            "generator_dropout",
            "generator_lr",
            "generator_residual",
            "generator_opt_betas",
            "discriminator_n_layers_hidden",
            "discriminator_n_units_hidden",
            "discriminator_activation",
            "discriminator_batch_norm",
            "discriminator_dropout",
            "discriminator_lr",
            "discriminator_opt_betas",
            "clipping_value",
            "lambda_gradient_penalty",
        ]

    @classmethod
    def get_metrics(cls) -> list[str]:
        return ["GLoss", "DLoss"]

    def generate(self, N: int, cond: Optional[np.ndarray] = None) -> np.ndarray:
        N = N or self.nrows
        self.generator.eval()

        condt: Optional[torch.Tensor] = None
        if cond is not None:
            condt = self._check_tensor(cond)
        with torch.no_grad():
            return self.metatransformer.inverse_apply(
                pd.DataFrame(self(N, condt).detach().cpu().numpy(), columns=self.columns)
            )

    def forward(
        self,
        N: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is None and self.n_units_conditional > 0:
            # sample from the original conditional
            if self._original_cond is None:
                raise ValueError("Invalid original conditional. Provide a valid value.")
            cond_idxs = torch.randint(len(self._original_cond), (N,))
            cond = self._original_cond[cond_idxs]

        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        if cond is not None and len(cond) != N:
            raise ValueError("cond length must match N")

        fixed_noise = torch.randn(N, self.generator_n_units_hidden, device=self.device)
        fixed_noise = self._append_optional_cond(fixed_noise, cond)

        return self.generator(fixed_noise)

    def _train_epoch_generator(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the G network
        self.generator.train()
        self.generator.optimizer.zero_grad()

        real_X_raw = X.to(self.device)
        real_X = self._append_optional_cond(real_X_raw, cond)
        batch_size = len(real_X)

        noise = torch.randn(batch_size, self.generator_n_units_hidden, device=self.device)
        noise = self._append_optional_cond(noise, cond)

        fake_raw = self.generator(noise)
        fake = self._append_optional_cond(fake_raw, cond)

        output = self.discriminator(fake).squeeze().float()
        # Calculate G's loss based on this output
        errG = -torch.mean(output)
        if hasattr(self, "generator_extra_penalty_cbks"):
            for extra_loss in self.generator_extra_penalty_cbks:
                errG += extra_loss(
                    real_X_raw,
                    fake_raw,
                    cond=cond,
                )

        # Calculate gradients for G
        errG.backward()

        # Update G
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clipping_value)
        self.generator.optimizer.step()

        if torch.isnan(errG):
            raise RuntimeError("NaNs detected in the generator loss")

        # Return loss
        return errG.item()

    def _train_epoch_discriminator(
        self,
        X: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> float:
        # Update the D network
        self.discriminator.train()

        errors = []

        batch_size = min(self.batch_size, len(X))

        # Train with all-real batch
        real_X = X.to(self.device)
        real_X = self._append_optional_cond(real_X, cond)

        real_labels = self.true_labels_generator(X).to(self.device).squeeze()
        real_output = self.discriminator(real_X).squeeze().float()

        # Train with all-fake batch
        noise = torch.randn(batch_size, self.generator_n_units_hidden, device=self.device)
        noise = self._append_optional_cond(noise, cond)

        fake_raw = self.generator(noise)
        fake = self._append_optional_cond(fake_raw, cond)

        fake_labels = self.fake_labels_generator(fake_raw).to(self.device).squeeze().float()
        fake_output = self.discriminator(fake.detach()).squeeze()

        # Compute errors. Some fake inputs might be marked as real for privacy guarantees.

        real_real_output = real_output[(real_labels * real_output) != 0]
        real_fake_output = fake_output[(fake_labels * fake_output) != 0]
        errD_real = torch.mean(torch.concat((real_real_output, real_fake_output)))

        fake_real_output = real_output[((1 - real_labels) * real_output) != 0]
        fake_fake_output = fake_output[((1 - fake_labels) * fake_output) != 0]
        errD_fake = torch.mean(torch.concat((fake_real_output, fake_fake_output)))

        penalty = self._loss_gradient_penalty(
            real_samples=real_X,
            fake_samples=fake,
            batch_size=batch_size,
        )
        errD = -errD_real + errD_fake

        self.discriminator.optimizer.zero_grad()
        if isinstance(self, DPMixin):
            # Adversarial loss
            # 1. split fwd-bkwd on fake and real images into two explicit blocks.
            # 2. no need to compute per_sample_gardients on fake data, disable hooks.
            # 3. re-enable hooks to obtain per_sample_gardients for real data.
            # fake fwd-bkwd
            self.discriminator.disable_hooks()
            penalty.backward(retain_graph=True)
            errD_fake.backward(retain_graph=True)

            self.discriminator.enable_hooks()
            errD_real.backward()  # HACK: calling bkwd without zero_grad() accumulates param gradients
        else:
            penalty.backward(retain_graph=True)
            errD.backward()

        # Update D
        if self.clipping_value > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clipping_value)
        self.discriminator.optimizer.step()

        errors.append(errD.item())

        if np.isnan(np.mean(errors)):
            raise RuntimeError("NaNs detected in the discriminator loss")

        return np.mean(errors)

    def _train_epoch(self) -> Tuple[float, float]:
        for data in tqdm(self.data_loader, desc="Batches", position=len(self.stats_bars) + 1, leave=False):
            cond: Optional[torch.Tensor] = None
            if self.n_units_conditional > 0:
                X, cond = data
            else:
                X = data[0]

            losses = {
                "DLoss": self._train_epoch_discriminator(X, cond),
                "GLoss": self._train_epoch_generator(X, cond),
            }
            self._record_metrics(losses)

        return np.mean(self.metrics["GLoss"][-len(self.data_loader) :]), np.mean(
            self.metrics["DLoss"][-len(self.data_loader) :]
        )

    def train(
        self,
        num_epochs: int = 100,
        patience: int = 5,
        displayed_metrics: list[str] = ["GLoss", "DLoss"],
    ) -> tuple[int, dict[str, np.ndarray]]:
        self._start_training(num_epochs, patience, displayed_metrics)

        for epoch in tqdm(range(num_epochs), desc="Epochs", position=len(self.stats_bars), leave=False):
            losses = self._train_epoch()
            if self._check_patience(epoch, losses[0]) and self._check_patience(epoch, losses[1]):
                num_epochs = epoch + 1
                break

        self._finish_training(num_epochs)
        return (num_epochs, self.metrics)

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _loss_gradient_penalty(
        self,
        real_samples: torch.tensor,
        fake_samples: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand([batch_size, 1]).to(self.device)
        # Get random interpolation between real and fake samples
        interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated).squeeze()
        labels = torch.ones((len(interpolated),), device=self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        return self.lambda_gradient_penalty * gradient_penalty

    def _append_optional_cond(self, X: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return X

        return torch.cat([X, cond], dim=1)
