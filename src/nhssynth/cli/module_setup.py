"""Specify all CLI-accessible modules and their configurations, the pipeline to run by default, and define special functions for the `config` and `pipeline` CLI option trees."""
import argparse
from typing import Callable, Final, Optional

from nhssynth.cli.common_arguments import COMMON_PARSERS
from nhssynth.cli.module_arguments import *
from nhssynth.modules import (
    dashboard,
    dataloader,
    evaluation,
    model,
    plotting,
    structure,
)


class ModuleConfig:
    """
    Represents a module's configuration, containing the following attributes:

    Attributes:
        func: A callable that executes the module's functionality.
        add_args: A callable that populates the module's sub-parser arguments.
        description: A description of the module's functionality.
        help: A help message for the module's command-line interface.
        common_parsers: A list of common parsers to add to the module's sub-parser, appending the 'dataset' and 'core' parsers to those passed.
    """

    def __init__(
        self,
        func: Callable[..., argparse.Namespace],
        add_args: Callable[..., None],
        description: str,
        help: str,
        common_parsers: Optional[list[str]] = None,
        no_seed: bool = False,
    ) -> None:
        self.func = func
        self.add_args = add_args
        self.description = description
        self.help = help
        self.common_parsers = ["core", "seed"] if not no_seed else ["core"]
        if common_parsers:
            assert set(common_parsers) <= COMMON_PARSERS.keys(), "Invalid common parser(s) specified."
            # merge the below two assert statements
            assert (
                "core" not in common_parsers and "seed" not in common_parsers
            ), "The 'seed' and 'core' parser groups are automatically added to all modules, remove the from `ModuleConfig`s."
            self.common_parsers += common_parsers

    def __call__(self, args: argparse.Namespace) -> argparse.Namespace:
        return self.func(args)


def run_pipeline(args: argparse.Namespace) -> None:
    """Runs the specified pipeline of modules with the passed configuration `args`."""
    print("Running full pipeline...")
    args.modules_to_run = PIPELINE
    for module_name in PIPELINE:
        args = MODULE_MAP[module_name](args)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to `parser` for each module in the pipeline."""
    for module_name in PIPELINE:
        MODULE_MAP[module_name].add_args(parser, f"{module_name} options")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to `parser` relating to configuration file handling and module-specific config overrides."""
    parser.add_argument(
        "-c",
        "--input-config",
        required=True,
        help="specify the config file name",
    )
    parser.add_argument(
        "-cp",
        "--custom-pipeline",
        action="store_true",
        help="infer a custom pipeline running order of modules from the config",
    )
    for module_name in PIPELINE:
        MODULE_MAP[module_name].add_args(parser, f"{module_name} option overrides", overrides=True)
    for module_name in VALID_MODULES - set(PIPELINE):
        MODULE_MAP[module_name].add_args(parser, f"{module_name} options overrides", overrides=True)


### EDIT BELOW HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

PIPELINE: Final = [
    "dataloader",
    "model",
    "evaluation",
    "dashboard",
]  # NOTE this determines the order of a pipeline run

MODULE_MAP: Final = {
    "dataloader": ModuleConfig(
        func=dataloader.run,
        add_args=add_dataloader_args,
        description="run the data loader module, to prepare the chosen dataset for use in other modules",
        help="prepare the dataset",
        common_parsers=["metadata", "typed", "transformed", "metatransformer", "sdv_metadata"],
    ),
    "structure": ModuleConfig(
        func=structure.run,
        add_args=add_structure_args,
        description="run the structural discovery module, to learn a structural model for use in training and evaluation",
        help="discover structure",
    ),
    "model": ModuleConfig(
        func=model.run,
        add_args=add_model_args,
        description="run the model architecture module, to train a synthetic data generator",
        help="train a model",
        common_parsers=["transformed", "metatransformer", "experiments", "synthetic_datasets", "model"],
    ),
    "evaluation": ModuleConfig(
        func=evaluation.run,
        add_args=add_evaluation_args,
        description="run the evaluation module, to evaluate an experiment",
        help="evaluate an experiment",
        common_parsers=["sdv_metadata", "typed", "experiments", "synthetic_datasets", "evaluations"],
    ),
    "plotting": ModuleConfig(
        func=plotting.run,
        add_args=add_plotting_args,
        description="run the plotting module, to generate plots for a given model and / or evaluation",
        help="generate plots",
        common_parsers=["typed", "evaluations"],
    ),
    "dashboard": ModuleConfig(
        func=dashboard.run,
        add_args=add_dashboard_args,
        description="run the dashboard module, to produce a streamlit dashboard",
        help="start up a streamlit dashboard to view the results of an evaluation",
        common_parsers=["typed", "experiments", "synthetic_datasets", "evaluations"],
        no_seed=True,
    ),
    "pipeline": ModuleConfig(
        func=run_pipeline,
        add_args=add_pipeline_args,
        description="run the full pipeline.",
        help="run the full pipeline",
    ),
    "config": ModuleConfig(
        func=None,
        add_args=add_config_args,
        description="run module(s) according to configuration specified by a file in `config/`; note that you can override parts of the configuration on the fly by using the usual CLI flags",
        help="run module(s) in line with a passed configuration file",
    ),
}

### EDIT ABOVE HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

VALID_MODULES = {x for x in MODULE_MAP.keys() if x not in {"pipeline", "config"}}

assert (
    set(PIPELINE) <= VALID_MODULES
), f"Invalid `PIPELINE` specification, must only contain valid modules from `MODULE_MAP`: {str(VALID_MODULES)}"


def get_parent_parsers(name: str, module_parsers: list[str]) -> list[argparse.ArgumentParser]:
    """Get a list of parent parsers for a given module, based on the module's `common_parsers` attribute."""
    if name in {"pipeline", "config"}:
        return [p(name == "config") for p in COMMON_PARSERS.values()]
    elif name == "dashboard":
        return [COMMON_PARSERS[pn](True) for pn in module_parsers]
    else:
        return [COMMON_PARSERS[pn]() for pn in module_parsers]


def add_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    module_config: ModuleConfig,
) -> argparse.ArgumentParser:
    """
    Add a subparser to an argparse argument parser.

    Args:
        subparsers: The subparsers action to which the subparser will be added.
        name: The name of the subparser.
        module_config: A [`ModuleConfig`][nhssynth.cli.module_setup.ModuleConfig] object containing information about the subparser, including a function to execute and a function to add arguments.

    Returns:
        The newly created subparser.
    """
    parent_parsers = get_parent_parsers(name, module_config.common_parsers)
    parser = subparsers.add_parser(
        name=name,
        description=module_config.description,
        help=module_config.help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parent_parsers,
    )
    if name not in {"pipeline", "config"}:
        module_config.add_args(parser, f"{name} options")
    else:
        module_config.add_args(parser)
    parser.set_defaults(func=module_config.func)
    return parser
