import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.io import *
from nhssynth.modules.dataloader.metadata import load_metadata
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def run(args: argparse.Namespace) -> argparse.Namespace:
    """
    Runs the main workflow of the dataloader module, transforms the dataset and writes the output and transformer used to disk.

    Args:
        args: An argparse Namespace containing the command line arguments.
    """
    print("Running dataloader module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    dir_input, fn_dataset, fn_metadata = check_input_paths(args.dataset, args.metadata, args.data_dir)

    # Load the dataset and accompanying metadata
    dataset = pd.read_csv(dir_input / fn_dataset, index_col=args.index_col)
    metadata = load_metadata(dir_input / fn_metadata, dataset)

    mt = MetaTransformer(metadata, args.sdv_workflow, args.allow_null_transformers, args.synthesizer)
    typed_dataset, prepared_dataset = mt.apply(dataset)

    write_data_outputs(typed_dataset, prepared_dataset, mt, fn_dataset, fn_metadata, dir_experiment, args)

    if "model" in args.modules_to_run:
        args.model_input = {
            "fn_dataset": fn_dataset,
            "prepared_dataset": prepared_dataset,
            "metatransformer": mt,
        }
    if "evaluation" in args.modules_to_run:
        args.evaluation_input = {"typed_data": typed_dataset}

    return args
