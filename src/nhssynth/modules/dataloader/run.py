import argparse

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.io import *
from nhssynth.modules.dataloader.metadata import load_metadata
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running dataloader module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    dir_input, fn_dataset, fn_metadata = check_input_paths(args.dataset, args.metadata, args.data_dir)

    dataset = pd.read_csv(dir_input / fn_dataset, index_col=args.index_col)
    metadata = load_metadata(dir_input / fn_metadata, dataset)

    mt = MetaTransformer(metadata, args.allow_null_transformers, args.synthesizer)
    typed_dataset, prepared_dataset = mt.apply(dataset)

    write_data_outputs(typed_dataset, prepared_dataset, mt, fn_dataset, fn_metadata, dir_experiment, args)

    if "model" in args.modules_to_run:
        args.module_handover.update(
            {
                "fn_dataset": fn_dataset,
                "prepared_dataset": prepared_dataset,
                "metatransformer": mt,
            }
        )
    if "evaluation" in args.modules_to_run:
        args.module_handover.update({"typed_dataset": typed_dataset, "sdtypes": mt.get_sdtypes()})

    print("")

    return args
