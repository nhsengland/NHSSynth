import argparse
import pickle
from pathlib import Path

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
    fn_input, fn_metadata = consistent_endings([(fn_input, ".csv"), (fn_metadata, ".yaml")])
    dir_data = Path(dir_data)
    fn_metadata = potential_suffix(fn_metadata, fn_input)
    warn_if_path_supplied([fn_input, fn_metadata], dir_data)
    check_exists([fn_input], dir_data)
    return dir_data, fn_input, fn_metadata


def check_output_paths(
    fn_input: str,
    fn_typed: str,
    fn_transformed: str,
    fn_transformer: str,
    dir_experiment: Path,
) -> tuple[str, str, str]:
    """
    Formats the output filenames for an experiment.

    Args:
        fn_input: The input data filename.
        fn_typed: The typed input data filename/suffix to append to `fn_input`.
        fn_transformed: The output data filename/suffix to append to `fn_input`.
        fn_transformer: The transformer filename/suffix to append to `fn_input`.
        dir_experiment: The experiment directory to write the outputs to.

    Returns:
        A tuple containing the formatted output filenames.

    Warnings:
        Raises a UserWarning when the path to `fn_transformed` includes directory separators, as this is not supported and may not work as intended.
        Raises a UserWarning when the path to `fn_transformer` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_typed, fn_transformed, fn_transformer = consistent_endings([fn_typed, fn_transformed, fn_transformer])
    fn_typed, fn_transformed, fn_transformer = potential_suffixes([fn_typed, fn_transformed, fn_transformer], fn_input)
    warn_if_path_supplied([fn_typed, fn_transformed, fn_transformer], dir_experiment)
    return fn_typed, fn_transformed, fn_transformer


def write_data_outputs(
    metatransformer: MetaTransformer,
    fn_dataset: str,
    fn_metadata: str,
    dir_experiment: Path,
    args: argparse.Namespace,
) -> None:
    """
    Writes the transformed data and metatransformer to disk.

    Args:
        metatransformer: The metatransformer used to transform the data into its transformed state.
        fn_dataset: The base dataset filename.
        fn_metadata: The metadata filename.
        dir_experiment: The experiment directory to write the outputs to.
        args: The parsed command line arguments.
    """
    fn_typed, fn_transformed, fn_transformer = check_output_paths(
        fn_dataset, args.typed, args.transformed, args.metatransformer, dir_experiment
    )
    metatransformer.save_metadata(dir_experiment / fn_metadata, args.collapse_yaml)
    metatransformer.get_typed_dataset().to_pickle(dir_experiment / fn_typed)
    metatransformer.get_transformed_dataset().to_pickle(dir_experiment / fn_transformed)
    metatransformer.get_transformed_dataset().to_csv(dir_experiment / (fn_transformed[:-3] + "csv"), index=False)
    with open(dir_experiment / fn_transformer, "wb") as f:
        pickle.dump(metatransformer, f)
