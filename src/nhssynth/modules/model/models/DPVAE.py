from typing import Optional

from nhssynth.modules.model.common.DPMixin import DPMixin
from nhssynth.modules.model.models.VAE import VAE
from opacus import GradSampleModule


class DPVAE(DPMixin, VAE):
    """
    A differentially private VAE. Accepts [`VAE`][nhssynth.modules.model.models.VAE.VAE] arguments
    as well as [`DPMixin`][nhssynth.modules.model.common.DPMixin.DPMixin] arguments.
    """

    def __init__(
        self,
        *args,
        target_epsilon: float = 3.0,
        target_delta: Optional[float] = None,
        max_grad_norm: float = 5.0,
        secure_mode: bool = False,
        shared_optimizer: bool = False,
        **kwargs,
    ) -> None:
        super(DPVAE, self).__init__(
            *args,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            secure_mode=secure_mode,
            # TODO fix shared_optimizer workflow for DP models
            shared_optimizer=False,
            **kwargs,
        )

    def make_private(self, num_epochs: int) -> GradSampleModule:
        """
        Make the [`Decoder`][nhssynth.modules.model.models.VAE.Decoder] differentially private
        unless `shared_optimizer` is True, in which case the whole VAE will be privatised.

        Args:
            num_epochs: The number of epochs to train for
        """
        if self.shared_optimizer:
            super().make_private(num_epochs)
        else:
            self.decoder = super().make_private(num_epochs, self.decoder)

    @classmethod
    def _get_args(cls) -> list[str]:
        return VAE._get_args() + DPMixin._get_args()
