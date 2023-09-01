import argparse
from typing import Any

import pandas as pd
from nhssynth.common import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from nhssynth.modules.model.io import load_required_data, write_data_outputs
from nhssynth.modules.model.models import MODELS
from nhssynth.modules.model.utils import get_experiments


def run_iter(
    experiment: dict[str, Any],
    architecture: str,
    real_dataset: pd.DataFrame,
    metatransformer: MetaTransformer,
    displayed_metrics: list[str],
    num_samples: int,
) -> pd.DataFrame:
    set_seed(experiment["seed"])
    model = MODELS[architecture](real_dataset, metatransformer, **experiment["model_config"])
    _, _ = model.train(
        **experiment["train_config"],
        displayed_metrics=displayed_metrics.copy(),
    )
    dataset = model.generate(num_samples)
    return {"dataset": dataset}, {"model": model}


def run(args: argparse.Namespace) -> argparse.Namespace:
    print("Running model architecture module...")

    set_seed(args.seed)
    dir_experiment = experiment_io(args.experiment_name)

    fn_dataset, real_dataset, metatransformer = load_required_data(args, dir_experiment)
    experiments = get_experiments(args)

    models = pd.DataFrame(index=experiments.index, columns=["model"])
    synthetic_datasets = pd.DataFrame(index=experiments.index, columns=["dataset"])

    for i, experiment in experiments.iterrows():
        print(
            f"\nRunning the {i[0]} architecture, repeat {i[1]} of {args.repeats}, with configuration {i[2]} of {experiment['num_configs']} 🤖\033[31m"
        )
        synthetic_datasets.loc[i], models.loc[i] = run_iter(
            experiment, i[0], real_dataset, metatransformer, args.displayed_metrics.copy(), args.num_samples
        )

    write_data_outputs(experiments, synthetic_datasets, models, fn_dataset, dir_experiment, args)

    if "dashboard" in args.modules_to_run or "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"fn_dataset": fn_dataset})
    if "evaluation" in args.modules_to_run or "plotting" in args.modules_to_run:
        args.module_handover.update({"synthetic_datasets": synthetic_datasets})

    print("")

    return args
