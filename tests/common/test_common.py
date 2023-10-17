import random

import numpy as np
import torch

from nhssynth.common import set_seed


def test_set_seed_int() -> None:
    seed_val = 123
    set_seed(seed_val)
    # Check if the seed for all three packages is set to the input value
    assert np.random.get_state()[1][0] == seed_val
    assert torch.initial_seed() == seed_val
    x = [random.random() for i in range(100)]
    set_seed(seed_val)
    y = [random.random() for i in range(100)]
    assert x == y
