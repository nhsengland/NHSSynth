import argparse
import pickle
from pathlib import Path

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metadata import output_metadata
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def check_input_paths(
    fn_input: str,
    fn_metadata: str,
    dir_data: str,
) -> tuple[Path, str, str]:
    """
    Formats the input filenames and directory for an experiment.

    Args:
        fn_input: The input data filename.
        fn_metadata: The metadata filename / suffix to append to `fn_input`.
        dir_data: The directory that should contain both of the above.

    Returns:
        A tuple containing the correct directory path, input data filename and metadata filename (used for both in and out).

    Warnings:
        Raises a UserWarning when the path to `fn_input` includes directory separators, as this is not supported and may not work as intended.
        Raises a UserWarning when the path to `fn_metadata` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_input, fn_metadata = consistent_endings([(fn_input, ".csv"), (fn_metadata, ".yaml")])
    dir_data = Path(dir_data)
    fn_metadata = potential_suffix(fn_metadata, fn_input)
    warn_if_path_supplied([fn_input, fn_metadata], dir_data)
    check_exists([fn_input], dir_data)
    return dir_data, fn_input, fn_metadata


def check_output_paths(
    fn_input: str,
    fn_typed: str,
    fn_prepared: str,
    fn_transformer: str,
    dir_experiment: Path,
) -> tuple[str, str, str]:
    """
    Formats the output filenames for an experiment.

    Args:
        fn_input: The input data filename.
        fn_typed: The typed input data filename/suffix to append to `fn_input`.
        fn_prepared: The output data filename/suffix to append to `fn_input`.
        fn_transformer: The transformer filename/suffix to append to `fn_input`.
        dir_experiment: The experiment directory to write the outputs to.

    Returns:
        A tuple containing the formatted output filenames.

    Warnings:
        Raises a UserWarning when the path to `fn_prepared` includes directory separators, as this is not supported and may not work as intended.
        Raises a UserWarning when the path to `fn_transformer` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_typed, fn_prepared, fn_transformer = consistent_endings([fn_typed, fn_prepared, fn_transformer])
    fn_typed, fn_prepared, fn_transformer = potential_suffixes([fn_typed, fn_prepared, fn_transformer], fn_input)
    warn_if_path_supplied([fn_typed, fn_prepared, fn_transformer], dir_experiment)
    return fn_typed, fn_prepared, fn_transformer


def write_data_outputs(
    typed_dataset: pd.DataFrame,
    prepared_dataset: pd.DataFrame,
    metatransformer: MetaTransformer,
    fn_dataset: str,
    fn_metadata: str,
    dir_experiment: Path,
    args: argparse.Namespace,
) -> None:
    """
    Writes the transformed data and metatransformer to disk.

    Args:
        typed_dataset: The typed version of the input dataset.
        prepared_dataset: The prepared version of the input dataset.
        metatransformer: The metatransformer used to transform the data into its prepared state.
        fn_dataset: The base dataset filename.
        fn_metadata: The metadata filename.
        dir_experiment: The experiment directory to write the outputs to.
        args: The parsed command line arguments.
    """
    fn_typed, fn_prepared, fn_transformer = check_output_paths(
        fn_dataset, args.typed, args.prepared, args.metatransformer, dir_experiment
    )
    output_metadata(dir_experiment / fn_metadata, metatransformer.get_assembled_metadata(), args.collapse_yaml)
    typed_dataset.to_pickle(dir_experiment / fn_typed)
    prepared_dataset.to_pickle(dir_experiment / fn_prepared)
    prepared_dataset.to_csv(dir_experiment / (fn_prepared[:-3] + "csv"), index=False)
    with open(dir_experiment / fn_transformer, "wb") as f:
        pickle.dump(metatransformer, f)
