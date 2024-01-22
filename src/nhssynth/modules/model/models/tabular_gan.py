from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from nhssynth.modules.model.common.samplers import ConditionalDatasetSampler
from nhssynth.modules.model.models.gan import GAN


class TabularGAN(GAN):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_gan.TabularGAN
        :parts: 1


    GAN for tabular data.

    This class combines GAN and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Reference dataset, used for training the tabular encoder
        n_units_latent: int
            Number of latent units
        cond: Optional
            Optional conditional
        generator_n_layers_hidden: int
            Number of hidden layers in the generator
        generator_n_units_hidden: int
            Number of hidden units in each layer of the Generator
        generator_activation: string, default 'elu'
            activationearity to use in the generator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
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
        generator_extra_penalties: list
            Additional penalties for the generator. Values: "identifiability_penalty"
        generator_extra_penalty_cbks: List[Callable]
            Additional loss callabacks for the generator. Used by the TabularGAN for the conditional loss
        discriminator_n_layers_hidden: int
            Number of hidden layers in the discriminator
        discriminator_n_units_hidden: int
            Number of hidden units in each layer of the discriminator
        discriminator_activation: string, default 'relu'
            activationearity to use in the discriminator. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        discriminator_n_iter: int
            Maximum number of iterations in the discriminator.
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
        batch_size: int
            Batch size
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        random_state: int
            random_state used
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        clipping_value: int, default 0
            Gradients clipping value. Zero disables the feature
        lambda_gradient_penalty: float = 10
            Weight for the gradient penalty
        lambda_identifiability_penalty: float = 0.1
            Weight for the identifiability penalty, if enabled
        dataloader_sampler: Optional[sampler.Sampler]
            Optional sampler for the dataloader, useful for conditional sampling
        device: Any = DEVICE
            CUDA/CPU
        adjust_inference_sampling: bool
            Adjust the marginal probabilities in the synthetic data to closer match the training set. Active only with the ConditionalSampler
        # privacy settings
        dp_enabled: bool
            Train the discriminator with Differential Privacy guarantees
        dp_delta: Optional[float]
            Optional DP delta: the probability of information accidentally being leaked. Usually 1 / len(dataset)
        dp_epsilon: float = 3
            DP epsilon: privacy budget, which is a measure of the amount of privacy that is preserved by a given algorithm. Epsilon is a number that represents the maximum amount of information that an adversary can learn about an individual from the output of a differentially private algorithm. The smaller the value of epsilon, the more private the algorithm is. For example, an algorithm with an epsilon of 0.1 preserves more privacy than an algorithm with an epsilon of 1.0.
        dp_max_grad_norm: float
            max grad norm used for gradient clipping
        dp_secure_mode: bool = False,
             if True uses noise generation approach robust to floating point arithmetic attacks.

        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        encoder:
            Pre-trained tabular encoder. If None, a new encoder is trained.
        encoder_whitelist:
            Ignore columns from encoding
    """

    def __init__(
        self,
        *args,
        generator_activation_out_discrete: str = "softmax",
        generator_activation_out_continuous: str = "none",
        **kwargs,
    ) -> None:
        self.generator_activation_out_discrete = generator_activation_out_discrete
        self.generator_activation_out_continuous = generator_activation_out_continuous

        super(TabularGAN, self).__init__(*args, **kwargs)

        def _generator_cond_loss(
            real_samples: torch.tensor,
            fake_samples: torch.Tensor,
            cond: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cond is None:
                return 0

            losses = []

            idx = 0
            cond_idx = 0

            for item in self.encoder.layout():
                length = item.output_dimensions

                if item.feature_type != "discrete":
                    idx += length
                    continue

                # create activate feature mask
                mask = cond[:, cond_idx : cond_idx + length].sum(axis=1).bool()

                if mask.sum() == 0:
                    idx += length
                    continue

                if not (fake_samples[mask, idx : idx + length] >= 0).all():
                    raise RuntimeError(f"Invalid samples after softmax = {fake_samples[mask, idx : idx + length]}")
                # fake_samples are after the Softmax activation
                # we filter active features in the mask
                item_loss = torch.nn.NLLLoss()(
                    torch.log(fake_samples[mask, idx : idx + length] + 1e-8),
                    torch.argmax(real_samples[mask, idx : idx + length], dim=1),
                )
                losses.append(item_loss)

                cond_idx += length
                idx += length

            if idx != real_samples.shape[1]:
                raise ValueError(f"Invalid offset idx = {idx}; real_samples.shape = {real_samples.shape}")

            if len(losses) == 0:
                return 0

            loss = torch.stack(losses, dim=-1)

            return loss.sum() / len(real_samples)

        self.generator_extra_penalty_cbks = [_generator_cond_loss]

    @classmethod
    def get_args(cls) -> list[str]:
        return GAN.get_args() + [
            "generator_activation_out_discrete",
            "generator_activation_out_continuous",
        ]

    @classmethod
    def get_metrics(cls) -> list[str]:
        return GAN.get_metrics()

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    # def get_encoder(self) -> TabularEncoder:
    #     return self.encoder

    def fit(
        self,
        X: pd.DataFrame,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        encoded: bool = False,
    ) -> Any:
        # preprocessing
        if encoded:
            X_enc = X
        else:
            X_enc = self.encode(X)

        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.get_dataset_conditionals()

        if cond is not None:
            if len(cond) != len(X_enc):
                raise ValueError(f"Invalid conditional shape. {cond.shape} expected {len(X_enc)}")

        # post processing
        self.adjust_inference_sampling(self._adjust_inference_sampling)

        return self

    def adjust_inference_sampling(self, enabled: bool) -> None:
        if self.predefined_conditional or self.dataloader_sampler is None:
            return

        self._adjust_inference_sampling = enabled

        if enabled:
            real_prob = self.dataloader_sampler.conditional_probs()
            sample_prob = self._extract_sample_prob()

            self.sample_prob = self._find_sample_p(real_prob, sample_prob)
        else:
            self.sample_prob = None

    def generate(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        samples = self(count, cond)
        return self.decode(pd.DataFrame(samples))

    def forward(self, count: int, cond: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> torch.Tensor:
        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.sample_conditional(count, p=self.sample_prob)

        return self.model.generate(count, cond=cond)

    def _extract_sample_prob(self) -> Optional[np.ndarray]:
        if self.predefined_conditional or self.dataloader_sampler is None:
            return None

        if self.dataloader_sampler.conditional_dimension() == 0:
            return None

        prob_list = list()
        batch_size = 10000

        for c in range(self.dataloader_sampler.conditional_dimension()):
            cond = self.dataloader_sampler.sample_conditional_for_class(batch_size, c)
            if cond is None:
                continue

            data_cond = self.model.generate(batch_size, cond=cond)

            syn_dataloader_sampler = ConditionalDatasetSampler(
                pd.DataFrame(data_cond),
                self.encoder.layout(),
            )

            prob = syn_dataloader_sampler.conditional_probs()
            prob_list.append(prob)

        prob_mat = np.stack(prob_list, axis=-1)

        return prob_mat

    def _find_sample_p(self, prob_real: Optional[np.ndarray], prob_mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if prob_real is None or prob_mat is None:
            return None

        def kl(alpha: np.ndarray, prob_real: np.ndarray, prob_mat: np.ndarray) -> np.ndarray:
            # alpha: _n_categories

            # f1: same as prob_real
            alpha_tensor = alpha[None, None, :]
            f1 = logsumexp(alpha_tensor, axis=-1, b=prob_mat)
            f2 = logsumexp(alpha)
            ce = -np.sum(prob_real * f1, axis=1) + f2
            return np.mean(ce)

        try:
            res = minimize(kl, np.ones(prob_mat.shape[-1]), (prob_real, prob_mat))
        except Exception:
            return np.ones(prob_mat.shape[-1]) / prob_mat.shape[-1]

        return np.exp(res.x) / np.sum(np.exp(res.x))
