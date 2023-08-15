import argparse
from pathlib import Path

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from nhssynth.modules.model import MODELS
from nhssynth.modules.model.io import load_required_data, output_full, output_iter


def run_iter(
    args: argparse.Namespace,
    fn_dataset: str,
    transformed_dataset: pd.DataFrame,
    metatransformer: MetaTransformer,
    dir_experiment: Path,
    architecture: str,
    iter_id: Optional[str] = None,
) -> pd.DataFrame:
    model = MODELS[architecture].from_args(args, transformed_dataset, metatransformer)
    _, _ = model.train(
        num_epochs=args.num_epochs,
        patience=args.patience,
        displayed_metrics=args.displayed_metrics.copy(),
    )
    synthetic = model.generate(args.num_samples)
    output_iter(
        model,
        synthetic,
        fn_dataset,
        args.synthetic,
        args.model_file,
        dir_experiment,
        architecture if not iter_id else f"{architecture}_{iter_id}",
    )
    return synthetic


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    experiment_bundle = {architecture: {} for architecture in args.architecture}

    req = load_required_data(args, dir_experiment)

    for architecture, architecture_bundle in experiment_bundle.items():
        if args.repeats > 1:
            for i in range(args.repeats):
                print(f"\nModel architecture: {architecture}\nRepeat: {i + 1} of {args.repeats}")
                set_seed(args.seed + i if args.seed else None)
                iter_id = str(args.seed + i) if args.seed else f"random_{str(i)}"
                architecture_bundle[iter_id] = {"data": run_iter(args, *req, dir_experiment, architecture, iter_id)}
        else:
            print(f"\nModel architecture: {architecture}")
            architecture_bundle = {"data": run_iter(args, *req, dir_experiment, architecture)}

    output_full(experiment_bundle, req[0], args.experiment_bundle, dir_experiment)

    if "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": req[0], "experiment_bundle": experiment_bundle})

    print("")

    return args
