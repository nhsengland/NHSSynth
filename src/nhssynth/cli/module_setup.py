import argparse
from typing import Any, Callable, Final

from nhssynth.cli.module_arguments import *
from nhssynth.modules import dataloader, evaluation, model, plotting, structure


class ModuleConfig:
    def __init__(
        self,
        func: Callable[..., Any],
        add_args_func: Callable[..., Any],
        description: str,
        help: str,
    ) -> None:
        """
        Represents a module's configuration, containing the following attributes:

        Args:
            func: A callable that executes the module's functionality.
            add_args_func: A callable that populates the module's sub-parser arguments.
            description: A description of the module's functionality.
            help: A help message for the module's command-line interface.
        """
        self.func = func
        self.add_args_func = add_args_func
        self.description = description
        self.help = help


def run_pipeline(args: argparse.Namespace) -> None:
    """Runs the specified pipeline of modules with the passed configuration `args`."""
    print("Running full pipeline...")
    args.modules_to_run = PIPELINE
    for module_name in PIPELINE:
        args = MODULE_MAP[module_name].func(args)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to a `parser` for each module in the pipeline."""
    for module_name in PIPELINE:
        group = parser.add_argument_group(title=module_name)
        MODULE_MAP[module_name].add_args_func(group)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to a `parser` relating to configuration file handling and module-specific config overrides."""
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
    for module_name in VALID_MODULES:
        group = parser.add_argument_group(title=f"{module_name} overrides")
        MODULE_MAP[module_name].add_args_func(group, override=True)


### EDIT BELOW HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

PIPELINE: Final = [
    "dataloader",
    "model",
    "evaluation",
    "plotting",
]  # NOTE this determines the order of a pipeline run

MODULE_MAP: Final = {
    "dataloader": ModuleConfig(
        dataloader.run,
        add_dataloader_args,
        "Run the Data Loader module, to prepare data for use in other modules.",
        "prepare input data",
    ),
    "structure": ModuleConfig(
        structure.run,
        add_structure_args,
        "Run the Structural Discovery module, to learn a structural model for use in training and evaluation.",
        "discover structure",
    ),
    "model": ModuleConfig(
        model.run,
        add_model_args,
        "Run the Architecture module, to train a model.",
        "train a model",
    ),
    "evaluation": ModuleConfig(
        evaluation.run,
        add_evaluation_args,
        "Run the Evaluation module, to evaluate an experiment.",
        "evaluate an experiment",
    ),
    "plotting": ModuleConfig(
        plotting.run,
        add_plotting_args,
        "Run the Plotting module, to generate plots for a given model and / or evaluation.",
        "generate plots",
    ),
    "pipeline": ModuleConfig(
        run_pipeline,
        add_pipeline_args,
        "Run the full pipeline.",
        "run the full pipeline",
    ),
    "config": ModuleConfig(
        None,
        add_config_args,
        "Run module(s) according to configuration specified by a file in `config/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
        "run module(s) in line with a passed configuration file",
    ),
}

### EDIT ABOVE HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

VALID_MODULES = {x for x in MODULE_MAP.keys() if x not in {"pipeline", "config"}}

assert (
    set(PIPELINE) <= VALID_MODULES
), f"Invalid `PIPELINE` specification, must only contain valid modules from `MODULE_MAP`: {str(VALID_MODULES)}"


def add_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    config: ModuleConfig,
) -> argparse.ArgumentParser:
    """
    Add a subparser to an argparse argument parser.

    Args:
        subparsers: The subparsers action to which the subparser will be added.
        name: The name of the subparser.
        config: A ModuleConfig object containing information about the subparser, including a function to execute and a function to add arguments.

    Returns:
        The newly created subparser.
    """
    parser = subparsers.add_parser(
        name=name,
        description=config.description,
        help=config.help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config.add_args_func(parser)
    parser.set_defaults(func=config.func)
    return parser
