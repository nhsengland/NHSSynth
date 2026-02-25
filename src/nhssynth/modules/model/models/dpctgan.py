from typing import Optional

from opacus import GradSampleModule

from nhssynth.modules.model.common.dp import DPMixin
from nhssynth.modules.model.models.ctgan import CTGAN


class DPCTGAN(DPMixin, CTGAN):
    """
    A differentially private CTGAN. Accepts :class:`CTGAN` arguments
    as well as :class:`DPMixin` arguments.

    DP is applied to the discriminator only, as it is the sole component
    that processes real training data.
    """

    def __init__(
        self,
        *args,
        target_epsilon: float = 3.0,
        target_delta: Optional[float] = None,
        max_grad_norm: float = 5.0,
        secure_mode: bool = False,
        **kwargs,
    ) -> None:
        super(DPCTGAN, self).__init__(
            *args,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            secure_mode=secure_mode,
            **kwargs,
        )

    def make_private(self, num_epochs: int) -> GradSampleModule:
        """
        Make the discriminator differentially private.

        Only the discriminator is privatised since only it processes real training data.

        Args:
            num_epochs: The number of epochs to train for.
        """
        self.discriminator = super().make_private(num_epochs, self.discriminator)

    @classmethod
    def get_args(cls) -> list[str]:
        return CTGAN.get_args() + DPMixin.get_args()

    @classmethod
    def get_metrics(cls) -> list[str]:
        return CTGAN.get_metrics() + DPMixin.get_metrics()
