import argparse
import pickle
from pathlib import Path

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def check_input_paths(fn_base: str, fn_prepared: str, fn_metatransformer: str, dir_experiment: Path) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_data: The name of the data file.
        fn_metadata: The name of the metadata file.
        fn_metatransformer: The name of the metatransformer file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_base, fn_prepared, fn_metatransformer = (
        consistent_ending(fn_base),
        consistent_ending(fn_prepared),
        consistent_ending(fn_metatransformer),
    )
    fn_prepared, fn_metatransformer = (
        potential_suffix(fn_prepared, fn_base),
        potential_suffix(fn_metatransformer, fn_base),
    )
    warn_if_path_supplied([fn_base, fn_prepared, fn_metatransformer], dir_experiment)
    check_exists([fn_prepared, fn_metatransformer], dir_experiment)
    return fn_base, fn_prepared, fn_metatransformer


def check_output_paths(fn_base: Path, fn_out: str, fn_model: str, dir_experiment: Path) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_model: The name of the model file.
        dir_experiment: The path to the experiment output directory.

    Returns:
        The path to output the model.
    """
    fn_out, fn_model = consistent_ending(fn_out, ".csv"), consistent_ending(fn_model, ".pt")
    fn_out, fn_model = potential_suffix(fn_out, fn_base), potential_suffix(fn_model, fn_base)
    warn_if_path_supplied([fn_out, fn_model], dir_experiment)
    return fn_out, fn_model


def load_required_data(
    args: argparse.Namespace, dir_experiment: Path
) -> tuple[str, pd.DataFrame, dict[str, int], MetaTransformer]:
    """
    Loads the data from `args` or from disk when the dataloader has not be run previously.

    Args:
        args: The arguments passed to the module, in this case potentially carrying the outputs of the dataloader module.
        dir_experiment: The path to the experiment directory.

    Returns:
        The data, metadata and metatransformer.
    """
    if getattr(args, "dataloader_output", None):
        return (
            args.dataloader_output["fn_real_data"],
            args.dataloader_output["data"],
            args.dataloader_output["metatransformer"],
        )
    else:
        if not args.real_data:
            raise ValueError(
                "You must provide `--real-data` when running this module on its own, please provide this (a prepared version and corresponding MetaTransformer must also exist in {dir_experiment})"
            )
        fn_real_data, fn_prepared_data, fn_metatransformer = check_input_paths(
            args.real_data, args.prepared_data, args.real_metatransformer, dir_experiment
        )

        with open(dir_experiment / fn_prepared_data, "rb") as f:
            data = pickle.load(f)
        with open(dir_experiment / fn_metatransformer, "rb") as f:
            metatransformer = pickle.load(f)

        return fn_real_data, data, metatransformer
