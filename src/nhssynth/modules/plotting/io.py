import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from nhssynth.common.io import *


def check_input_paths(
    fn_dataset: str, fn_typed: str, fn_synthetic: str, fn_report: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_typed: The name of the typed data file.
        fn_synthetic: The name of the metatransformer file.
        fn_report: The name of the report file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset, fn_typed, fn_synthetic, fn_report = consistent_endings([fn_dataset, fn_typed, fn_synthetic, fn_report])
    fn_typed, fn_synthetic, fn_report = potential_suffixes([fn_typed, fn_synthetic, fn_report], fn_dataset)
    warn_if_path_supplied([fn_dataset, fn_typed, fn_synthetic, fn_report], dir_experiment)
    check_exists([fn_typed, fn_synthetic], dir_experiment)
    return fn_dataset, fn_typed, fn_synthetic, fn_report


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
    if all(x in args.module_handover for x in ["fn_dataset", "typed_dataset", "synthetic", "report"]):
        return (
            args.module_handover["fn_dataset"],
            args.module_handover["typed_dataset"],
            args.module_handover["synthetic"],
            args.module_handover["report"],
        )
    else:
        fn_dataset, fn_typed, fn_synthetic, fn_report = check_input_paths(
            args.dataset, args.typed, args.synthetic, args.report, dir_experiment
        )

        with open(dir_experiment / fn_typed, "rb") as f:
            real_data = pickle.load(f)
        with open(dir_experiment / fn_synthetic, "rb") as f:
            synthetic_data = pickle.load(f)
        if (dir_experiment / fn_report).exists():
            with open(dir_experiment / fn_report, "rb") as f:
                report = pickle.load(f)
        else:
            report = None

        return fn_dataset, real_data, synthetic_data, report
