import pickle
from pathlib import Path

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def check_input_paths(
    fn_data: str,
    fn_metadata: str,
    dir_data: str,
) -> tuple[Path, str, str]:
    """
    Formats the input and output filenames and directories for an experiment.

    Args:
        fn_in: The input data filename.
        fn_metadata: The metadata filename / suffix to append to `fn_in`.
        dir_data: The directory that should contain both of the above.

    Returns:
        A tuple containing the formatted input, output, and metadata (in and out) paths.

    Warnings:
        Raises a UserWarning when the path to `fn_in` includes directory separators, as this is not supported and may not work as intended.
        Raises a UserWarning when the path to `fn_metadata` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_data, fn_metadata = consistent_ending(fn_data, ".csv"), consistent_ending(fn_metadata, ".yaml")
    dir_data = Path(dir_data)
    fn_metadata = potential_suffix(fn_metadata, fn_data)
    warn_if_path_supplied([fn_data, fn_metadata], dir_data)
    check_exists([fn_data], dir_data)
    return dir_data, fn_data, fn_metadata


def check_output_paths(
    fn_in: str,
    fn_out: str,
    fn_transformer: str,
    dir_experiment: Path,
) -> tuple[str, str]:
    """
    Formats the output filenames and directories for an experiment.

    Args:
        fn_in: The input data filename.
        fn_out: The output data filename / suffix to append to `fn_in`.
        fn_transformer: The transformer filename / suffix to append to `fn_in`.

    Returns:
        A tuple containing the formatted output and transformer (in and out) paths.

    Warnings:
        Raises a UserWarning when the path to `fn_out` includes directory separators, as this is not supported and may not work as intended.
        Raises a UserWarning when the path to `fn_transformer` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_out, fn_transformer = consistent_ending(fn_out), consistent_ending(fn_transformer)
    fn_out, fn_transformer = potential_suffix(fn_out, fn_in), potential_suffix(fn_transformer, fn_in)
    warn_if_path_supplied([fn_out, fn_transformer], dir_experiment)
    return fn_out, fn_transformer


def write_data_outputs(
    transformed_input: pd.DataFrame,
    metatransformer: MetaTransformer,
    fn_out: str,
    fn_transformer: str,
    dir_experiment: Path,
) -> None:
    """
    Writes the transformed data and metatransformer to disk.

    Args:
        transformed_input: The prepared version of the input data.
        metatransformer: The metatransformer used to transform the data into its prepared state.
        fn_out: The filename to dump the prepared data to.
        fn_transformer: The filename to dump the metatransformer to.
        dir_experiment: The experiment directory to write the outputs to.
    """
    transformed_input.to_pickle(dir_experiment / fn_out)
    transformed_input.to_csv(dir_experiment / (fn_out[:-3] + "csv"), index=False)
    with open(dir_experiment / fn_transformer, "wb") as f:
        pickle.dump(metatransformer, f)
