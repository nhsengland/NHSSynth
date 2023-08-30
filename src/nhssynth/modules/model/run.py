import argparse
from typing import Any

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from nhssynth.modules.model.io import load_required_data, output_full, output_iter
from nhssynth.modules.model.models import MODELS
from nhssynth.modules.model.utils import get_experiments


def run_iter(
    experiment: dict[str, Any],
    real_dataset: pd.DataFrame,
    metatransformer: MetaTransformer,
    patience: int,
    displayed_metrics: list[str],
    num_samples: int,
) -> pd.DataFrame:
    set_seed(experiment["seed"])
    model = MODELS[experiment["architecture"]](real_dataset, metatransformer, **experiment["model_config"])
    _, _ = model.train(
        num_epochs=experiment["num_epochs"],
        patience=patience,
        displayed_metrics=displayed_metrics.copy(),
    )
    synthetic = model.generate(num_samples)
    return model, synthetic


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_dataset, metatransformer = load_required_data(args, dir_experiment)
    experiments = get_experiments(args)

    for experiment in experiments:
        print(
            f"\nRunning the {experiment['architecture']} architecture with configuration {experiment['config_idx']} of {experiment['num_configs']}, repeat {experiment['repeat']} of {args.repeats} 🤖"
        )
        model, synthetic_dataset = run_iter(
            experiment, real_dataset, metatransformer, args.patience, args.displayed_metrics.copy(), args.num_samples
        )
        output_iter(
            model,
            synthetic_dataset,
            fn_dataset,
            args.synthetic,
            args.model_file,
            dir_experiment,
            experiment["id"],
        )
        experiment["dataset"] = synthetic_dataset

    output_full(experiments, fn_dataset, args.experiments, dir_experiment)

    if "dashboard" in args.modules_to_run or "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset})
    if "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"experiments": experiments})

    print("")

    return args
