from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def experiment_dir(tmp_path) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    return experiment_dir


@pytest.fixture
def fn_transformed() -> str:
    return "transformed"


@pytest.fixture(autouse=True)
def transformed(experiment_dir, fn_transformed) -> pd.DataFrame:
    transformed = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    transformed.to_pickle(experiment_dir / (fn_transformed + ".pkl"))
    return transformed
