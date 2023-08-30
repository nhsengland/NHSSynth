import argparse
import itertools
from typing import Any, Union

import pandas as pd
from nhssynth.modules.dataloader.metatransformer import MetaTransformer
from nhssynth.modules.model import MODELS


def wrap_arg(arg) -> Union[list, tuple]:
    if not isinstance(arg, list) and not isinstance(arg, tuple):
        return [arg]
    return arg


def get_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments = []
    for arch_name, repeat in itertools.product(*[wrap_arg(args.architecture), list(range(args.repeats))]):
        arch = MODELS[arch_name]
        model_args = {
            arg: wrap_arg(getattr(args, arg)) for arg in arch.get_args() + ["batch_size", "use_gpu", "num_epochs"]
        }
        model_configs = list(itertools.product(*model_args.values()))
        for i, values in enumerate(model_configs):
            model_config = {k: v for k, v in zip(model_args.keys(), values) if v is not None}
            num_epochs = model_config.pop("num_epochs")
            experiment = {
                "architecture": arch_name,
                "model_config": model_config,
                "config_idx": str(i + 1),
                "num_configs": len(model_configs),
                "seed": args.seed + repeat if args.seed else None,
                "repeat": str(repeat + 1),
                "num_epochs": num_epochs,
            }
            experiment["id"] = (
                arch_name
                + (f"_config_{experiment['config_idx']}" if len(model_configs) > 1 else "")
                + (f"_repeat_{experiment['repeat']}" if args.repeats > 1 else "")
            )
            experiments.append(experiment)
    return experiments
