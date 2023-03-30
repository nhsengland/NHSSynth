import argparse

import numpy as np
import pandas as pd
from nhssynth.modules.dataloader.io import setup_io
from nhssynth.modules.dataloader.metadata import (
    instantiate_dtypes,
    load_metadata,
    output_metadata,
)
from nhssynth.modules.dataloader.transformers import (
    apply_transformer,
    instantiate_metatransformer,
)


def run(args: argparse.Namespace) -> None:
    """
    Runs the main workflow of the dataloader module, transforming the input data and writing the output to file.

    Args:
        args: An argparse Namespace containing the command line arguments.

    Returns:
        None
    """

    print("Running dataloader...")

    if args.seed:
        np.random.seed(args.seed)

    input_path, output_path, metadata_input_path, metadata_output_path = setup_io(
        args.input_file, args.output_file, args.metadata_file, args.dir, args.run_name
    )

    # Load the dataset and accompanying metadata
    input = pd.read_csv(input_path, index_col=args.index_col)
    metadata = load_metadata(metadata_input_path, input)

    # Setup the input data dtypes and apply them
    dtypes = instantiate_dtypes(metadata, input)
    typed_input = input.astype(dtypes)

    # Setup the metatransformer
    metatransformer = instantiate_metatransformer(
        metadata, typed_input, args.sdv_workflow, args.allow_null_transformers
    )

    # Output the metadata corresponding to `transformed_input`, for reproducibility
    output_metadata(metadata_output_path, dtypes, metatransformer, args.sdv_workflow, args.collapse_yaml)

    print("Transforming input...")
    transformed_input = apply_transformer(metatransformer, typed_input, args.sdv_workflow)

    # Write the transformed input to the appropriate file
    print("Writing output")
    transformed_input.to_csv(output_path, index=False)
