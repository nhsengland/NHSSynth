import argparse

from nhssynth.cli.module_arguments import *
from nhssynth.modules import dataloader, evaluation, model, plotting, structure


class ModuleConfig:
    def __init__(self, func, args_func, description: str, help: str):
        self.func = func
        self.args_func = args_func
        self.description = description
        self.help = help


def run_pipeline_generator(pipeline: list):
    def run_pipeline(args: argparse.Namespace):
        print("Running full pipeline...")
        for module_name in pipeline:
            MODULE_MAP[module_name].func(args)

    return run_pipeline


def pipeline_args_generator(pipeline: list):
    def add_pipeline_args(parser: argparse.ArgumentParser):
        for module_name in pipeline:
            group = parser.add_argument_group(title=module_name)
            MODULE_MAP[module_name].args_func(group)

    return add_pipeline_args


def config_args_generator(pipeline: list):
    def add_config_args(parser: argparse.ArgumentParser):
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
        for module_name in pipeline:
            if module_name not in ["pipeline", "config"]:
                group = parser.add_argument_group(title=f"{module_name} overrides")
                MODULE_MAP[module_name].args_func(group, override=True)

    return add_config_args


### EDIT BELOW HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

PIPELINE = ["dataloader", "model", "evaluation", "plotting"]

MODULE_MAP = {
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
        run_pipeline_generator(PIPELINE),
        pipeline_args_generator(PIPELINE),
        "Run the full pipeline.",
        "run the full pipeline",
    ),
    "config": ModuleConfig(
        None,
        config_args_generator(PIPELINE),
        "Run module(s) according to configuration specified by a file in `config/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
        "run module(s) in line with a passed configuration file",
    ),
}

### EDIT ABOVE HERE TO ADD MODULES / ALTER PIPELINE BEHAVIOUR

VALID_MODULES = {x for x in MODULE_MAP.keys() if x not in {"pipeline", "config"}}

assert (
    set(PIPELINE) <= VALID_MODULES
), f"Invalid `PIPELINE` specification, must only contain valid modules from `MODULE_MAP`: {str(VALID_MODULES)}"


def add_subparser(subparsers: argparse._SubParsersAction, name: str, config: ModuleConfig):
    parser = subparsers.add_parser(
        name=name,
        description=config.description,
        help=config.help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config.args_func(parser)
    parser.set_defaults(func=config.func)
    return parser
