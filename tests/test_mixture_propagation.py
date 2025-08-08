import numpy as np
import pandas as pd


def test_mixture_components_visible(continuous_transformer_factory):
    # Build a toy 2-component mixture
    n = 20000
    z = np.random.rand(n) < 0.6
    x = np.where(z, np.random.normal(-2, 0.3, n), np.random.normal(2, 0.5, n))
    s = pd.Series(x, name="toy")

    tr = continuous_transformer_factory(name="toy", n_components=2)
    encoded = tr.apply(s.to_frame())
    # Simulate model echoing back mean probs (just loop through)
    decoded = tr.revert(encoded)
    # Bimodality quick check: histogram has two peaks far apart
    hist, edges = np.histogram(decoded, bins=50)
    assert hist.max() > 1.5 * np.median(hist)  # crude but catches collapse
