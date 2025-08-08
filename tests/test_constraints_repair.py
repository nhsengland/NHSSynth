import pandas as pd
import numpy as np


def test_unary_bounds_repair(meta):
    # Assume fixture 'meta' builds MetaTransformer with constraints on 'x': [0, 1]
    df = pd.DataFrame({"x": np.random.normal(0, 2, size=5000)})
    out = meta.repair_constraints(df)
    assert out["x"].min() >= 0 - 1e-12
    assert out["x"].max() <= 1 + 1e-12


def test_binary_relation_repair(meta):
    df = pd.DataFrame(
        {"a": np.random.normal(0, 1, 5000), "b": np.random.normal(0, 1, 5000)}
    )
    # meta.constraints includes a <= b
    out = meta.repair_constraints(df)
    assert (out["a"] <= out["b"]).all()


def test_sum_relation_repair(meta):
    df = pd.DataFrame(
        {
            "a": np.abs(np.random.gamma(2.0, 2.0, 5000)),
            "b": np.abs(np.random.gamma(2.0, 2.0, 5000)),
            "c": np.abs(np.random.gamma(2.0, 2.0, 5000)),
        }
    )
    # meta.constraints includes a + b <= c
    out = meta.repair_constraints(df)
    assert (out["a"] + out["b"] <= out["c"] + 1e-12).all()
