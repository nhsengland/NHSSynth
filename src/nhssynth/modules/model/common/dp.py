import warnings
from abc import ABC
from typing import Optional

import numpy as np
import torch.nn as nn
from nhssynth.modules.model.common.model import Model
from opacus import GradSampleModule, PrivacyEngine


class DPMixin(ABC):
    """
    Mixin class to make a [`Model`][nhssynth.modules.model.common.model.Model] differentially private

    Args:
        target_epsilon: The target epsilon for the model during training
        target_delta: The target delta for the model during training
        max_grad_norm: The maximum norm for the gradients, they are trimmed to this norm if they are larger
        secure_mode: Whether to use the 'secure mode' of PyTorch's DP-SGD implementation via the `csprng` package

    Attributes:
        target_epsilon: The target epsilon for the model during training
        target_delta: The target delta for the model during training
        max_grad_norm: The maximum norm for the gradients, they are trimmed to this norm if they are larger
        secure_mode: Whether to use the 'secure mode' of PyTorch's DP-SGD implementation via the `csprng` package

    Raises:
        TypeError: If the inheritor is not a `Model`
    """

    def __init__(
        self,
        *args,
        target_epsilon: float = 3.0,
        target_delta: Optional[float] = None,
        max_grad_norm: float = 5.0,
        secure_mode: bool = False,
        **kwargs,
    ):
        if not isinstance(self, Model):
            raise TypeError("DPMixin can only be used with Model classes")
        super(DPMixin, self).__init__(*args, **kwargs)
        self.target_epsilon: float = target_epsilon
        self.target_delta: float = target_delta or 1 / self.nrows
        self.max_grad_norm: float = max_grad_norm
        self.secure_mode: bool = secure_mode

    def make_private(self, num_epochs: int, module: Optional[nn.Module] = None) -> GradSampleModule:
        """
        Make the passed module (or the full model if a module is not passed), and its associated optimizer and data loader private.

        Args:
            num_epochs: The number of epochs to train for, used to calculate the privacy budget.
            module: The module to make private.

        Returns:
            The privatised module.
        """
        module = module or self
        self.privacy_engine = PrivacyEngine(secure_mode=self.secure_mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
            module, module.optim, self.data_loader = self.privacy_engine.make_private_with_epsilon(
                module=module,
                optimizer=module.optim,
                data_loader=self.data_loader,
                epochs=num_epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
            )
        print(
            f"Using sigma={module.optim.noise_multiplier} and C={self.max_grad_norm} to target (ε, δ) = ({self.target_epsilon}, {self.target_delta})-differential privacy.".format()
        )
        self.get_epsilon = self.privacy_engine.accountant.get_epsilon
        return module

    def _generate_metric_str(self, key) -> str:
        """Generates a string to display the current value of the metric `key`."""
        if key == "Privacy":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in log")
                warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
                val = self.get_epsilon(self.target_delta)
            self.metrics[key] = np.append(self.metrics[key], val)
            return f"{(key + ' ε Spent:').ljust(self.max_length)}  {val:.4f}"
        else:
            return super()._generate_metric_str(key)

    def get_args() -> list[str]:
        return ["target_epsilon", "target_delta", "max_grad_norm", "secure_mode"]

    def _start_training(self, num_epochs, patience, displayed_metrics):
        self.make_private(num_epochs)
        super()._start_training(num_epochs, patience, displayed_metrics)
