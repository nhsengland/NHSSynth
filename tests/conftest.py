from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def experiment_dir(tmp_path) -> Path:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    return experiment_dir


@pytest.fixture
def fn_prepared() -> str:
    return "prepared"


@pytest.fixture(autouse=True)
def prepared(experiment_dir, fn_prepared) -> pd.DataFrame:
    prepared = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    prepared.to_pickle(experiment_dir / (fn_prepared + ".pkl"))
    return prepared
