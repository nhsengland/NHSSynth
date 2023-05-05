import warnings
from pathlib import Path

import pytest
from nhssynth.common.io import *


@pytest.fixture
def experiments_dir(tmp_path) -> Path:
    """
    Create a temporary directory for experiments and return its path.
    """
    dir_experiments = tmp_path / "experiments"
    dir_experiments.mkdir()
    return dir_experiments


def test_experiment_io_creates_dir_and_returns_path(experiments_dir) -> None:
    experiment_name = "my_experiment"
    expected_path = experiments_dir / experiment_name

    actual_path = experiment_io(experiment_name, experiments_dir)

    assert actual_path == expected_path
    assert actual_path.exists()


def test_experiment_io_creates_nested_dir_and_returns_path(experiments_dir) -> None:
    experiment_name = "nested/experiment"
    expected_path = experiments_dir / experiment_name

    actual_path = experiment_io(experiment_name, experiments_dir)

    assert actual_path == expected_path
    assert actual_path.exists()


def test_consistent_endings_all_combinations() -> None:
    args = ["file1.pkl", "file2", ("file3", ".yaml"), ("file4", ".csv", "_processed"), ("/dir/file5", ".pkl")]
    expected_endings = ["file1.pkl", "file2.pkl", "file3.yaml", "file4_processed.csv", "/dir/file5.pkl"]

    actual_endings = consistent_endings(args)

    assert actual_endings == expected_endings


def test_potential_suffixes() -> None:
    fns = ["not_suffix.csv", "_suffix.yaml"]
    fn_base = "file.pkl"
    expected_fns = ["not_suffix.csv", "file_suffix.yaml"]

    actual_fns = potential_suffixes(fns, fn_base)

    assert actual_fns == expected_fns


def test_check_exists_file_exists(tmp_path) -> None:
    file_path1 = tmp_path / "test2.txt"
    file_path1.touch()
    file_path2 = tmp_path / "test1.txt"
    file_path2.touch()

    check_exists(["test1.txt", "test2.txt"], tmp_path)


def test_check_exists_file_does_not_exist(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        check_exists(["test.txt"], tmp_path)


def test_warn_if_path_supplied_no_warnings(tmp_path) -> None:
    file_path = tmp_path / "test.txt"
    file_path.touch()

    # check if no warnings are raised
    with warnings.catch_warnings(record=True) as w:
        warn_if_path_supplied(["test.txt"], tmp_path)
        assert len(w) == 0


def test_warn_if_path_supplied_with_warnings(tmp_path) -> None:
    subdir_path = tmp_path / "subdir"
    subdir_path.mkdir()
    file_path = subdir_path / "test.txt"
    file_path.touch()

    # check if warnings are raised
    with warnings.catch_warnings(record=True) as w:
        warn_if_path_supplied(["subdir/test.txt"], tmp_path)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "attempting to read data from" in str(w[-1].message)
