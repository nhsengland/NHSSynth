import argparse

import pytest
from nhssynth.modules.model.io import *


@pytest.fixture
def experiment_dir(tmp_path) -> str:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    return experiment_dir


@pytest.fixture
def fn_dataset():
    return "dataset"


@pytest.fixture
def fn_prepared():
    return "prepared"


@pytest.fixture
def fn_metatransformer():
    return "_metatransformer"


@pytest.fixture
def fn_synthetic():
    return "_synthetic"


@pytest.fixture
def fn_model():
    return "_model"


@pytest.fixture
def args_no_handover(fn_dataset, fn_prepared, fn_metatransformer) -> argparse.Namespace:
    args = argparse.Namespace()
    args.dataset = fn_dataset
    args.prepared = fn_prepared
    args.metatransformer = fn_metatransformer
    return args


@pytest.fixture
def args_handover(fn_dataset, fn_prepared, fn_metatransformer) -> argparse.Namespace:
    args = argparse.Namespace()
    args.module_handover = {
        "dataset": fn_dataset,
        "prepared": fn_prepared,
        "metatransformer": fn_metatransformer,
    }
    return args


def test_check_input_paths(experiment_dir, fn_dataset, fn_prepared, fn_metatransformer):
    expected_input_paths = (fn_dataset + ".pkl", fn_prepared + ".pkl", fn_dataset + fn_metatransformer + ".pkl")

    experiment_dir.joinpath(fn_prepared + ".pkl").touch()
    experiment_dir.joinpath(fn_dataset + fn_metatransformer + ".pkl").touch()
    input_paths = check_input_paths(fn_dataset, fn_prepared, fn_metatransformer, experiment_dir)

    assert input_paths == expected_input_paths


def test_check_input_paths_with_invalid_filenames(experiment_dir, fn_dataset) -> None:
    with pytest.raises(FileNotFoundError):
        check_input_paths(fn_dataset, "not_prepared.pkl", "not_metatransformer.pkl", experiment_dir)


def test_check_input_paths_with_nested_dir(experiment_dir, fn_dataset, fn_prepared, fn_metatransformer) -> None:
    nested_dir = experiment_dir / "prepared"
    nested_dir.mkdir()
    nested_dir.joinpath(fn_prepared + ".pkl").touch()
    experiment_dir.joinpath(fn_dataset + fn_metatransformer + ".pkl").touch()

    with pytest.warns(UserWarning, match="Using the path supplied appended to"):
        check_input_paths(fn_dataset, "prepared/" + fn_prepared, fn_metatransformer, experiment_dir)


def test_check_output_paths(experiment_dir, fn_dataset, fn_synthetic, fn_model) -> None:
    expected_output_paths = (fn_dataset + fn_synthetic + "_DPVAE.pkl", fn_dataset + fn_model + "_DPVAE.pt")

    output_paths = check_output_paths(fn_dataset, fn_synthetic, fn_model, experiment_dir, "DPVAE")

    assert output_paths == expected_output_paths


def test_check_output_paths_with_seed(experiment_dir, fn_dataset, fn_synthetic, fn_model) -> None:
    expected_output_paths = (fn_dataset + fn_synthetic + "_DPVAE_123.pkl", fn_dataset + fn_model + "_DPVAE_123.pt")

    output_paths = check_output_paths(fn_dataset, fn_synthetic, fn_model, experiment_dir, "DPVAE", seed=123)

    assert output_paths == expected_output_paths


def test_load_required_data_no_handover(args_no_handover, experiment_dir) -> None:
    experiment_dir.joinpath(args_no_handover.prepared + ".pkl").touch()
    experiment_dir.joinpath(args_no_handover.dataset + args_no_handover.metatransformer + ".pkl").touch()

    fn_dataset, prepared_dataset, mt = load_required_data(args_no_handover, experiment_dir)

    assert fn_dataset == args_no_handover.dataset
    assert prepared_dataset == args_no_handover.prepared
    assert mt == args_no_handover.metatransformer


def test_load_required_data_no_handover_with_invalid_filenames(args_no_handover, experiment_dir) -> None:
    args_no_handover.prepared = "not_prepared"
    args_no_handover.metatransformer = "not_metatransformer"

    with pytest.raises(FileNotFoundError):
        load_required_data(args_no_handover, experiment_dir)


def test_load_required_data_from_args(args_handover, experiment_dir) -> None:
    experiment_dir.joinpath(args_handover.module_handover["prepared"] + ".pkl").touch()
    experiment_dir.joinpath(
        args_handover.module_handover["dataset"] + args_handover.module_handover["metatransformer"] + ".pkl"
    ).touch()

    fn_dataset, prepared_dataset, mt = load_required_data(args_handover, experiment_dir)

    assert fn_dataset == args_handover.module_handover["dataset"]
    assert prepared_dataset == args_handover.module_handover["prepared"]
    assert mt == args_handover.module_handover["metatransformer"]
