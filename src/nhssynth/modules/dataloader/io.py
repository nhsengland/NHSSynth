import pickle
from pathlib import Path

import pandas as pd
from nhssynth.common.io import *
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
    fn_input, fn_metadata = consistent_ending(fn_input, ".csv"), consistent_ending(fn_metadata, ".yaml")
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
    fn_typed, fn_prepared, fn_transformer = (
        consistent_ending(fn_typed),
        consistent_ending(fn_prepared),
        consistent_ending(fn_transformer),
    )
    fn_typed, fn_prepared, fn_transformer = (
        potential_suffix(fn_typed, fn_input),
        potential_suffix(fn_prepared, fn_input),
        potential_suffix(fn_transformer, fn_input),
    )
    warn_if_path_supplied([fn_typed, fn_prepared, fn_transformer], dir_experiment)
    return fn_typed, fn_prepared, fn_transformer


def write_data_outputs(
    prepared_input: pd.DataFrame,
    metatransformer: MetaTransformer,
    fn_prepared: str,
    fn_transformer: str,
    dir_experiment: Path,
) -> None:
    """
    Writes the transformed data and metatransformer to disk.

    Args:
        prepared_input: The prepared version of the input data.
        metatransformer: The metatransformer used to transform the data into its prepared state.
        fn_prepared: The filename to dump the prepared data to.
        fn_transformer: The filename to dump the metatransformer to.
        dir_experiment: The experiment directory to write the outputs to.
    """
    prepared_input.to_pickle(dir_experiment / fn_prepared)
    prepared_input.to_csv(dir_experiment / (fn_prepared[:-3] + "csv"), index=False)
    with open(dir_experiment / fn_transformer, "wb") as f:
        pickle.dump(metatransformer, f)
