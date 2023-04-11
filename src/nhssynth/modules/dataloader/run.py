import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.io import *
from nhssynth.modules.dataloader.metadata import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def run(args: argparse.Namespace) -> argparse.Namespace:
    """
    Runs the main workflow of the dataloader module, transforming the input data and writing the output and transformer used to file.

    Args:
        args: An argparse Namespace containing the command line arguments.
    """
    print("Running dataloader module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    dir_input, fn_input, fn_metadata = check_input_paths(args.input, args.metadata, args.data_dir)

    # Load the dataset and accompanying metadata
    input = pd.read_csv(dir_input / fn_input, index_col=args.index_col)
    metadata = load_metadata(dir_input / fn_metadata, input)

    mt = MetaTransformer(metadata, args.sdv_workflow, args.allow_null_transformers, args.synthesizer)
    typed_input, prepared_input = mt.apply(input)

    # Output the metadata corresponding to `prepared_input`, for reproducibility
    if not args.discard_metadata:
        output_metadata(dir_experiment / fn_metadata, mt.get_assembled_metadata(), args.collapse_yaml)

    # Write the transformed input to the appropriate file
    fn_output_typed, fn_output_prepared, fn_transformer = check_output_paths(
        fn_input, args.output, args.metatransformer, dir_experiment
    )
    if "model" not in args.modules_to_run or args.write_all:
        write_data_outputs(prepared_input, mt, fn_output_prepared, fn_transformer, dir_experiment)
    if "evaluation" not in args.modules_to_run or args.write_all:
        typed_input.to_pickle(dir_experiment / fn_output_typed)

    if "model" in args.modules_to_run:
        args.model_input = {
            "fn_base": fn_input,
            "prepared_data": prepared_input,
            "metatransformer": mt,
        }
    if "evaluation" in args.modules_to_run:
        args.evaluation_input = {"typed_data": typed_input}

    return args
