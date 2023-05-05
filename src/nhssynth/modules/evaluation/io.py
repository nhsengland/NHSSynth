import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metadata import get_sdtypes, load_metadata


def check_input_paths(
    fn_dataset: str, fn_typed: str, fn_synthetic: str, fn_metadata: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_typed: The name of the typed data file.
        fn_synthetic: The name of the metatransformer file.
        fn_metadata: The name of the metadata file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset, fn_typed, fn_synthetic, fn_metadata = consistent_endings(
        [fn_dataset, fn_typed, fn_synthetic, (fn_metadata, ".yaml")]
    )
    fn_typed, fn_synthetic, fn_metadata = potential_suffixes([fn_typed, fn_synthetic, fn_metadata], fn_dataset)
    warn_if_path_supplied([fn_dataset, fn_typed, fn_synthetic, fn_metadata], dir_experiment)
    check_exists([fn_typed, fn_synthetic, fn_metadata], dir_experiment)
    return fn_dataset, fn_typed, fn_synthetic, fn_metadata


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
    if all(x in args.module_handover for x in ["dataset", "typed", "synthetic", "sdtypes"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["typed"],
            args.module_handover["synthetic"],
            args.module_handover["sdtypes"],
        )
    else:
        fn_dataset, fn_typed, fn_synthetic, fn_metadata = check_input_paths(
            args.dataset, args.typed, args.synthetic, args.metadata, dir_experiment
        )

        with open(dir_experiment / fn_typed, "rb") as f:
            real_data = pickle.load(f)
        with open(dir_experiment / fn_synthetic, "rb") as f:
            synthetic_data = pickle.load(f)
        sdtypes = get_sdtypes(load_metadata(dir_experiment / fn_metadata, real_data))

        return fn_dataset, real_data, synthetic_data, sdtypes
