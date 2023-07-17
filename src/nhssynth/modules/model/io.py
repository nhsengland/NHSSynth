import argparse
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def check_input_paths(
    fn_dataset: str, fn_transformed: str, fn_metatransformer: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_transformed: The name of the transformed data file.
        fn_metatransformer: The name of the metatransformer file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset, fn_transformed, fn_metatransformer = consistent_endings(
        [fn_dataset, fn_transformed, fn_metatransformer]
    )
    fn_transformed, fn_metatransformer = potential_suffixes([fn_transformed, fn_metatransformer], fn_dataset)
    warn_if_path_supplied([fn_dataset, fn_transformed, fn_metatransformer], dir_experiment)
    check_exists([fn_transformed, fn_metatransformer], dir_experiment)
    return fn_dataset, fn_transformed, fn_metatransformer


def check_output_paths(
    fn_dataset: Path,
    fn_synthetic: str,
    fn_model: str,
    dir_experiment: Path,
    model: str,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_synthetic: The name of the synthetic data file.
        fn_model: The name of the model file.
        dir_experiment: The path to the experiment output directory.
        model: The name of the model used.
        seed: The seed used to generate the synthetic data.

    Returns:
        The path to output the model.
    """
    fn_synthetic, fn_model = consistent_endings(
        [
            (fn_synthetic, ".pkl", f"_{model}" + (f"_{str(seed)}" if seed else "")),
            (fn_model, ".pt", f"_{model}" + (f"_{str(seed)}" if seed else "")),
        ]
    )
    fn_synthetic, fn_model = potential_suffixes([fn_synthetic, fn_model], fn_dataset)
    warn_if_path_supplied([fn_synthetic, fn_model], dir_experiment)
    return fn_synthetic, fn_model


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
    if all(x in args.module_handover for x in ["dataset", "transformed", "metatransformer"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["transformed"],
            args.module_handover["metatransformer"],
        )
    else:
        fn_dataset, fn_transformed, fn_metatransformer = check_input_paths(
            args.dataset, args.transformed, args.metatransformer, dir_experiment
        )

        with open(dir_experiment / fn_transformed, "rb") as f:
            data = pickle.load(f)
        with open(dir_experiment / fn_metatransformer, "rb") as f:
            mt = pickle.load(f)

        return fn_dataset, data, mt
