import random

import numpy as np
import torch


def set_seed(seed: None | int = None) -> None:
    """
    Set the seed for numpy and torch.

    Args:
        seed: The seed to set.
    """
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
