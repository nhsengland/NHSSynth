import argparse

from nhssynth.cli.module_arguments import *
from nhssynth.modules import dataloader, evaluation, model, plotting, structure


class ModuleConfig:
    def __init__(self, func, args_func, description: str, help: str):
        self.func = func
        self.args_func = args_func
        self.description = description
        self.help = help


def run_pipeline(args: argparse.Namespace):
    print("Running full pipeline...")
    dataloader.run(args)
    structure.run(args)
    model.run(args)
    evaluation.run(args)
    plotting.run(args)


def config():
    raise "This function should never be called"


# TODO I wonder if it is possible to add a flag to ModuleConfig
# saying whether the module should be included in a full run,
# and construct the functions in module_arguments accordingly
MODULE_MAP = {
    "pipeline": ModuleConfig(
        run_pipeline,
        add_all_module_args,
        "Run an automatically configured module or set of modules specified by a config file in `config/`. Note that you can override parts of the configuration on the fly via the usual CLI flags.",
        "run the full pipeline",
    ),
    "config": ModuleConfig(
        config,
        add_config_args,
        "Run module(s) according to configuration specified by a file in `config/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
        "run module(s) in line with a passed configuration file",
    ),
    "prepare": ModuleConfig(
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
    "train": ModuleConfig(
        model.run,
        add_model_args,
        "Run the Architecture module, to train a model.",
        "train a model",
    ),
    "evaluate": ModuleConfig(
        evaluation.run,
        add_evaluation_args,
        "Run the Evaluation module, to evaluate an experiment.",
        "evaluate an experiment",
    ),
    "plot": ModuleConfig(
        plotting.run,
        add_plotting_args,
        "Run the Plotting module, to generate plots for a given model and / or evaluation.",
        "generate plots",
    ),
}
