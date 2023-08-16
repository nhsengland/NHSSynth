import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from nhssynth.common.io import *


def check_input_paths(
    fn_dataset: str, fn_typed: str, fn_experiment_bundle: str, fn_sdv_metadata: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_typed: The name of the typed data file.
        fn_experiment_bundle: The name of the metatransformer file.
        fn_sdv_metadata: The name of the SDV metadata file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset = Path(fn_dataset).stem
    fn_typed, fn_experiment_bundle, fn_sdv_metadata = consistent_endings(
        [fn_typed, fn_experiment_bundle, fn_sdv_metadata]
    )
    fn_typed, fn_experiment_bundle, fn_sdv_metadata = potential_suffixes(
        [fn_typed, fn_experiment_bundle, fn_sdv_metadata], fn_dataset
    )
    warn_if_path_supplied([fn_typed, fn_experiment_bundle, fn_sdv_metadata], dir_experiment)
    check_exists([fn_typed, fn_experiment_bundle, fn_sdv_metadata], dir_experiment)
    return fn_dataset, fn_typed, fn_experiment_bundle, fn_sdv_metadata


def output_eval(
    collected_evals: dict[str, pd.DataFrame],
    fn_dataset: Path,
    fn_evaluation_bundle: str,
    dir_experiment: Path,
):
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_evaluation_bundle: The name of the evaluation bundle file.
        dir_experiment: The path to the experiment output directory.

    Returns:
        The path to output the model.
    """
    fn_evaluation_bundle = consistent_ending(fn_evaluation_bundle)
    fn_evaluation_bundle = potential_suffix(fn_evaluation_bundle, fn_dataset)
    warn_if_path_supplied([fn_evaluation_bundle], dir_experiment)
    with open(dir_experiment / fn_evaluation_bundle, "wb") as f:
        pickle.dump(collected_evals, f)


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
    if all(x in args.module_handover for x in ["dataset", "typed", "experiment_bundle", "sdv_metadata"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["typed"],
            args.module_handover["experiment_bundle"],
            args.module_handover["sdv_metadata"],
        )
    else:
        fn_dataset, fn_typed, fn_experiment_bundle, fn_sdv_metadata = check_input_paths(
            args.dataset, args.typed, args.experiment_bundle, args.sdv_metadata, dir_experiment
        )
        with open(dir_experiment / fn_typed, "rb") as f:
            real_data = pickle.load(f)
        with open(dir_experiment / fn_sdv_metadata, "rb") as f:
            sdv_metadata = pickle.load(f)
        with open(dir_experiment / fn_experiment_bundle, "rb") as f:
            experiment_bundle = pickle.load(f)

        return fn_dataset, real_data, experiment_bundle, sdv_metadata
