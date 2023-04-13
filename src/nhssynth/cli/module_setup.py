import argparse
from typing import Any, Callable, Final

from nhssynth.cli.common_arguments import COMMON_PARSERS
from nhssynth.cli.module_arguments import *
from nhssynth.modules import dataloader, evaluation, model, plotting, structure


class ModuleConfig:
    def __init__(
        self,
        func: Callable[..., Any],
        add_args_func: Callable[..., Any],
        description: str,
        help: str,
        common_parsers: list[str] = None,
    ) -> None:
        """
        Represents a module's configuration, containing the following attributes:

        Args:
            func: A callable that executes the module's functionality.
            add_args_func: A callable that populates the module's sub-parser arguments.
            description: A description of the module's functionality.
            help: A help message for the module's command-line interface.
            common_parsers: A list of common parsers to add to the module's sub-parser.
        """
        self.func = func
        self.add_args_func = add_args_func
        self.description = description
        self.help = help
        if common_parsers:
            assert set(common_parsers) <= COMMON_PARSERS.keys(), "Invalid common parser(s) specified."
            assert (
                "dataset" not in common_parsers
            ), "The 'dataset' parser is automatically added to all modules, remove it from the ModuleConfig."
            assert (
                "core" not in common_parsers
            ), "The 'core' parser is automatically added to all modules, remove it from the ModuleConfig."
            self.common_parsers = ["dataset", "core"] + common_parsers
        else:
            self.common_parsers = ["dataset", "core"]


def run_pipeline(args: argparse.Namespace) -> None:
    """Runs the specified pipeline of modules with the passed configuration `args`."""
    print("Running full pipeline...")
    args.modules_to_run = PIPELINE
    for module_name in PIPELINE:
        args = MODULE_MAP[module_name].func(args)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments to a `parser` for each module in the pipeline."""
    for module_name in PIPELINE:
        MODULE_MAP[module_name].add_args_func(parser, f"{module_name} options")


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
        MODULE_MAP[module_name].add_args_func(parser, f"{module_name} option overrides", overrides=True)


### EDIT BELOW HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

PIPELINE: Final = [
    "dataloader",
    "model",
    "evaluation",
    "plotting",
]  # NOTE this determines the order of a pipeline run

MODULE_MAP: Final = {
    "dataloader": ModuleConfig(
        func=dataloader.run,
        add_args_func=add_dataloader_args,
        description="Run the Data Loader module, to prepare the chosen dataset for use in other modules.",
        help="prepare the dataset",
        common_parsers=["metadata", "typed", "prepared", "metatransformer"],
    ),
    "structure": ModuleConfig(
        func=structure.run,
        add_args_func=add_structure_args,
        description="Run the Structural Discovery module, to learn a structural model for use in training and evaluation.",
        help="discover structure",
    ),
    "model": ModuleConfig(
        func=model.run,
        add_args_func=add_model_args,
        description="Run the Model Architecture module, to train a synthetic data generator.",
        help="train a model",
        common_parsers=["prepared", "metatransformer", "synthetic"],
    ),
    "evaluation": ModuleConfig(
        func=evaluation.run,
        add_args_func=add_evaluation_args,
        description="Run the Evaluation module, to evaluate an experiment.",
        help="evaluate an experiment",
        common_parsers=["metadata", "typed", "synthetic", "report"],
    ),
    "plotting": ModuleConfig(
        func=plotting.run,
        add_args_func=add_plotting_args,
        description="Run the Plotting module, to generate plots for a given model and / or evaluation.",
        help="generate plots",
    ),
    "pipeline": ModuleConfig(
        func=run_pipeline,
        add_args_func=add_pipeline_args,
        description="Run the full pipeline.",
        help="run the full pipeline",
    ),
    "config": ModuleConfig(
        func=None,
        add_args_func=add_config_args,
        description="Run module(s) according to configuration specified by a file in `config/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
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
        module_config: A ModuleConfig object containing information about the subparser, including a function to execute and a function to add arguments.

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
        module_config.add_args_func(parser, f"{name} options")
    else:
        module_config.add_args_func(parser)
    parser.set_defaults(func=module_config.func)
    return parser
