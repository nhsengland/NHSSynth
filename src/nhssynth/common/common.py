"""Common functions for all modules."""
import random

import numpy as np
import torch


def set_seed(seed: None | int = None) -> None:
    """
    (Potentially) set the seed for numpy, torch and random.

    Args:
        seed: The seed to set.
    """
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
