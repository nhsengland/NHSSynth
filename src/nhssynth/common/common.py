"""Common functions for all modules."""
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> None:
    """
    (Potentially) set the seed for numpy, torch and random. If no seed is provided, nothing happens.

    Args:
        seed: The seed to set.
    """
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
