import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from nhssynth.common.io import *


class Evaluations:
    def __init__(self, evaluations: dict[str, dict[str, Any]]):
        self.contents = evaluations


def check_input_paths(
    fn_dataset: str, fn_typed: str, fn_synthetic_datasets: str, fn_sdv_metadata: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_typed: The name of the typed real dataset file.
        fn_synthetic_datasets: The filename of the collection of synethtic datasets.
        fn_sdv_metadata: The name of the SDV metadata file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset = Path(fn_dataset).stem
    fn_typed, fn_synthetic_datasets, fn_sdv_metadata = consistent_endings(
        [fn_typed, fn_synthetic_datasets, fn_sdv_metadata]
    )
    fn_typed, fn_synthetic_datasets, fn_sdv_metadata = potential_suffixes(
        [fn_typed, fn_synthetic_datasets, fn_sdv_metadata], fn_dataset
    )
    warn_if_path_supplied([fn_typed, fn_synthetic_datasets, fn_sdv_metadata], dir_experiment)
    check_exists([fn_typed, fn_synthetic_datasets, fn_sdv_metadata], dir_experiment)
    return fn_dataset, fn_typed, fn_synthetic_datasets, fn_sdv_metadata


def output_eval(
    evaluations: pd.DataFrame,
    fn_dataset: Path,
    fn_evaluations: str,
    dir_experiment: Path,
) -> None:
    """
    Sets up the input and output paths for the model files.

    Args:
        evaluations: The evaluations to output.
        fn_dataset: The base name of the dataset.
        fn_evaluations: The filename of the collection of evaluations.
        dir_experiment: The path to the experiment output directory.

    Returns:
        The path to output the model.
    """
    fn_evaluations = consistent_ending(fn_evaluations)
    fn_evaluations = potential_suffix(fn_evaluations, fn_dataset)
    warn_if_path_supplied([fn_evaluations], dir_experiment)
    with open(dir_experiment / fn_evaluations, "wb") as f:
        pickle.dump(Evaluations(evaluations), f)


def load_required_data(
    args: argparse.Namespace, dir_experiment: Path
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Loads the data from `args` or from disk when the dataloader has not be run previously.

    Args:
        args: The arguments passed to the module, in this case potentially carrying the outputs of the dataloader module.
        dir_experiment: The path to the experiment directory.

    Returns:
        The dataset name, the real data, the bundle of synthetic data from the modelling stage, and the SDV metadata.
    """
    if all(x in args.module_handover for x in ["dataset", "typed", "synthetic_datasets", "sdv_metadata"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["typed"],
            args.module_handover["synthetic_datasets"],
            args.module_handover["sdv_metadata"],
        )
    else:
        fn_dataset, fn_typed, fn_synthetic_datasets, fn_sdv_metadata = check_input_paths(
            args.dataset, args.typed, args.synthetic_datasets, args.sdv_metadata, dir_experiment
        )
        with open(dir_experiment / fn_typed, "rb") as f:
            real_data = pickle.load(f).contents
        with open(dir_experiment / fn_sdv_metadata, "rb") as f:
            sdv_metadata = pickle.load(f)
        with open(dir_experiment / fn_synthetic_datasets, "rb") as f:
            synthetic_datasets = pickle.load(f).contents

        return fn_dataset, real_data, synthetic_datasets, sdv_metadata
