import argparse
import itertools
from typing import Any, Union

import pandas as pd

from nhssynth.modules.model import MODELS


def wrap_arg(arg: Any) -> Union[list, tuple]:
    """
    Wraps a single argument in a list if it is not already a list or tuple.

    Args:
        arg: The argument to wrap.

    Returns:
        The wrapped argument.
    """
    if not isinstance(arg, list) and not isinstance(arg, tuple):
        return [arg]
    return arg


def configs_from_arg_combinations(args: argparse.Namespace, arg_list: list[str]) -> list[dict[str, Any]]:
    """
    Generates a list of configurations from a list of arguments. Each configuration is one of a cartesian product of
    the arguments provided and identified in `arg_list`.

    Args:
        args: The arguments.
        arg_list: The list of arguments to generate configurations from.

    Returns:
        A list of configurations.
    """
    wrapped_args = {arg: wrap_arg(getattr(args, arg)) for arg in arg_list}
    combinations = list(itertools.product(*wrapped_args.values()))
    return [{k: v for k, v in zip(wrapped_args.keys(), values) if v is not None} for values in combinations]


def get_experiments(args: argparse.Namespace) -> pd.DataFrame:
    """
    Generates a dataframe of experiments from the arguments provided.

    Args:
        args: The arguments.

    Returns:
        A dataframe of experiments indexed by architecture, repeat and config ID.
    """
    experiments = pd.DataFrame(
        columns=["architecture", "repeat", "config", "model_config", "seed", "train_config", "num_configs"]
    )
    train_configs = configs_from_arg_combinations(args, ["num_epochs", "patience"])
    for arch_name, repeat in itertools.product(*[wrap_arg(args.architecture), list(range(args.repeats))]):
        arch = MODELS[arch_name]
        model_configs = configs_from_arg_combinations(args, arch.get_args() + ["batch_size", "use_gpu"])
        for i, (train_config, model_config) in enumerate(itertools.product(train_configs, model_configs)):
            experiments.loc[len(experiments.index)] = {
                "architecture": arch_name,
                "repeat": repeat + 1,
                "config": i + 1,
                "model_config": model_config,
                "num_configs": len(model_configs) * len(train_configs),
                "seed": args.seed + repeat if args.seed else None,
                "train_config": train_config,
            }
    return experiments.set_index(["architecture", "repeat", "config"], drop=True)
