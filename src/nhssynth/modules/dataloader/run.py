import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.io import *
from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running dataloader module...\033[35m")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    dir_input, fn_dataset, fn_metadata = check_input_paths(args.dataset, args.metadata, args.data_dir)

    dataset = pd.read_csv(dir_input / fn_dataset, index_col=args.index_col)
    metadata = MetaData.from_path(dataset, dir_input / fn_metadata)
    mt = MetaTransformer(dataset, metadata, args.missingness, args.impute)
    mt.apply()

    fn_dataset = write_data_outputs(mt, fn_dataset, fn_metadata, dir_experiment, args)

    if "model" in args.modules_to_run:
        args.module_handover.update(
            {
                "dataset": fn_dataset,
                "transformed": mt.get_transformed_dataset(),
                "metatransformer": mt,
                "sdv_metadata": mt.get_sdv_metadata(),
            }
        )
    if "evaluation" in args.modules_to_run:
        args.module_handover.update({"typed": mt.get_typed_dataset()})

    print("\033[0m")

    return args
