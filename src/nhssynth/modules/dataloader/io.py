import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from tqdm import tqdm


class TypedDataset:
    def __init__(self, typed_dataset: pd.DataFrame):
        self.contents = typed_dataset


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
        UserWarning: When the path to `fn_input` includes directory separators, as this is not supported and may not work as intended.
        UserWarning: When the path to `fn_metadata` includes directory separators, as this is not supported and may not work as intended.
    """
    fn_input, fn_metadata = consistent_endings([(fn_input, ".csv"), (fn_metadata, ".yaml")])
    dir_data = Path(dir_data)
    fn_metadata = potential_suffix(fn_metadata, fn_input)
    warn_if_path_supplied([fn_input, fn_metadata], dir_data)
    check_exists([fn_input], dir_data)
    return dir_data, fn_input, fn_metadata


def check_output_paths(
    fn_dataset: str,
    fn_typed: str,
    fn_transformed: str,
    fn_transformer: str,
    fn_constraint_graph: str,
    fn_sdv_metadata: str,
    dir_experiment: Path,
) -> tuple[str, str, str]:
    """
    Formats the output filenames for an experiment.

    Args:
        fn_dataset: The input data filename.
        fn_typed: The typed input data filename/suffix to append to `fn_dataset`.
        fn_transformed: The output data filename/suffix to append to `fn_dataset`.
        fn_transformer: The metatransformer filename/suffix to append to `fn_dataset`.
        fn_constraint_graph: The constraint graph filename/suffix to append to `fn_dataset`.
        fn_sdv_metadata: The SDV metadata filename/suffix to append to `fn_dataset`.
        dir_experiment: The experiment directory to write the outputs to.

    Returns:
        A tuple containing the formatted output filenames.

    Warnings:
        UserWarning: When any of the filenames include directory separators, as this is not supported and may not work as intended.
    """
    fn_dataset = Path(fn_dataset).stem
    fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata = consistent_endings(
        [fn_typed, fn_transformed, fn_transformer, (fn_constraint_graph, ".html"), fn_sdv_metadata]
    )
    fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata = potential_suffixes(
        [fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata], fn_dataset
    )
    warn_if_path_supplied(
        [fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata], dir_experiment
    )
    return fn_dataset, fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata


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

    Returns:
        The filename of the dataset used.
    """
    fn_dataset, fn_typed, fn_transformed, fn_transformer, fn_constraint_graph, fn_sdv_metadata = check_output_paths(
        fn_dataset,
        args.typed,
        args.transformed,
        args.metatransformer,
        args.constraint_graph,
        args.sdv_metadata,
        dir_experiment,
    )
    metatransformer.save_metadata(dir_experiment / fn_metadata, args.collapse_yaml)
    metatransformer.save_constraint_graphs(dir_experiment / fn_constraint_graph)
    with open(dir_experiment / fn_typed, "wb") as f:
        pickle.dump(TypedDataset(metatransformer.get_typed_dataset()), f)
    transformed_dataset = metatransformer.get_transformed_dataset()
    transformed_dataset.to_pickle(dir_experiment / fn_transformed)
    if args.write_csv:
        chunks = np.array_split(transformed_dataset.index, 100)
        for chunk, subset in enumerate(tqdm(chunks, desc="Writing transformed dataset to CSV", unit="chunk")):
            if chunk == 0:
                transformed_dataset.loc[subset].to_csv(
                    dir_experiment / (fn_transformed[:-3] + "csv"), mode="w", index=False
                )
            else:
                transformed_dataset.loc[subset].to_csv(
                    dir_experiment / (fn_transformed[:-3] + "csv"), mode="a", index=False, header=False
                )
    with open(dir_experiment / fn_transformer, "wb") as f:
        pickle.dump(metatransformer, f)
    with open(dir_experiment / fn_sdv_metadata, "wb") as f:
        pickle.dump(metatransformer.get_sdv_metadata(), f)

    return fn_dataset
