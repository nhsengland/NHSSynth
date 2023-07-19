import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metadata import MetaData


def check_input_paths(
    fn_dataset: str, fn_typed: str, fn_experiment_bundle: str, fn_metadata: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_typed: The name of the typed data file.
        fn_experiment_bundle: The name of the metatransformer file.
        fn_metadata: The name of the metadata file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset, fn_typed, fn_experiment_bundle, fn_metadata = consistent_endings(
        [fn_dataset, fn_typed, fn_experiment_bundle, (fn_metadata, ".yaml")]
    )
    fn_typed, fn_experiment_bundle, fn_metadata = potential_suffixes(
        [fn_typed, fn_experiment_bundle, fn_metadata], fn_dataset
    )
    warn_if_path_supplied([fn_dataset, fn_typed, fn_experiment_bundle, fn_metadata], dir_experiment)
    check_exists([fn_typed, fn_experiment_bundle, fn_metadata], dir_experiment)
    return fn_dataset, fn_typed, fn_experiment_bundle, fn_metadata


def output_eval(
    eval_bundle: dict,
    fn_dataset: Path,
    fn_eval_bundle: str,
    dir_experiment: Path,
):
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_eval_bundle: The name of the evaluation bundle file.
        dir_experiment: The path to the experiment output directory.

    Returns:
        The path to output the model.
    """
    fn_eval_bundle = consistent_ending(fn_eval_bundle)
    fn_eval_bundle = potential_suffix(fn_eval_bundle, fn_dataset)
    warn_if_path_supplied([fn_eval_bundle], dir_experiment)
    with open(dir_experiment / fn_eval_bundle, "wb") as f:
        pickle.dump(eval_bundle, f)


def load_required_data(
    args: argparse.Namespace, dir_experiment: Path
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Loads the data from `args` or from disk when the dataloader has not be run previously.

    Args:
        args: The arguments passed to the module, in this case potentially carrying the outputs of the dataloader module.
        dir_experiment: The path to the experiment directory.

    Returns:
        The data, metadata and metatransformer.
    """
    if all(x in args.module_handover for x in ["dataset", "typed", "experiment_bundle", "metadata"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["typed"],
            args.module_handover["experiment_bundle"],
            args.module_handover["metadata"],
        )
    else:
        fn_dataset, fn_typed, fn_experiment_bundle, fn_metadata = check_input_paths(
            args.dataset, args.typed, args.experiment_bundle, args.metadata, dir_experiment
        )
        with open(dir_experiment / fn_typed, "rb") as f:
            real_data = pickle.load(f)
        sdmetadata = MetaData.load(dir_experiment / fn_metadata, real_data).get_sdmetadata()
        with open(dir_experiment / fn_experiment_bundle, "rb") as f:
            experiment_bundle = pickle.load(f)

        return fn_dataset, real_data, experiment_bundle, sdmetadata
