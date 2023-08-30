"""Read, write and process config files, including handling of module-specific / common config overrides."""
import argparse
import warnings
from importlib.metadata import version as ver
from typing import Any, Callable

import yaml
from nhssynth.cli.module_setup import MODULE_MAP, PIPELINE, run_pipeline
from nhssynth.common.dicts import *


def get_default_and_required_args(
    top_parser: argparse.ArgumentParser,
    module_parsers: dict[str, argparse.ArgumentParser],
) -> tuple[dict[str, Any], list[str]]:
    """
    Get the default and required arguments for the top-level parser and the current run's corresponding list of module parsers.

    Args:
        top_parser: The top-level parser (contains common arguments).
        module_parsers: The dict of module-level parsers mapped to their names.

    Returns:
        A tuple containing two elements:
            - A dictionary containing all arguments and their default values.
            - A list of key-value-pairs of the required arguments and their associated module.
    """
    all_actions = {"top-level": top_parser._actions} | {m: p._actions for m, p in module_parsers.items()}
    defaults = {}
    required_args = []
    for module, actions in all_actions.items():
        for action in actions:
            if action.dest not in ["help", "==SUPPRESS=="]:
                defaults[action.dest] = action.default
                if action.required:
                    required_args.append({"arg": action.dest, "module": module})
    return defaults, required_args


def read_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    all_subparsers: dict[str, argparse.ArgumentParser],
) -> argparse.Namespace:
    """
    Hierarchically assembles a config `argparse.Namespace` object for the inferred modules to run and execute, given a file.

    1. Load the YAML file containing the config to read from
    2. Check a valid `run_type` is specified or infer it and determine the list of `modules_to_run`
    3. Establish the appropriate default config from the `parser` and `all_subparsers` for the determined `modules_to_run`
    4. Overwrite this config with the specified (sub)set of config in the YAML file
    5. Overwrite again with passed command-line `args` (these are considered 'overrides')
    6. Run the appropriate module(s) or pipeline with the resulting configuration namespace object

    Args:
        args: Namespace object containing arguments from the command line
        parser: top-level `ArgumentParser` object containing common arguments
        all_subparsers: dictionary of `ArgumentParser` objects, one for each module

    Returns:
        A Namespace object containing the assembled configuration settings

    Raises:
        AssertionError: if any required arguments are missing from the configuration file / overrides
    """
    # Open the passed yaml file and load into a dictionary
    with open(f"config/{args.input_config}.yaml") as stream:
        config_dict = yaml.safe_load(stream)

    valid_run_types = [x for x in all_subparsers.keys() if x != "config"]

    version = config_dict.pop("version", None)
    if version and version != version("nhssynth"):
        warnings.warn(
            f"This config file's specified version ({version}) does not match the currently installed version of nhssynth ({version('nhssynth')}), results may differ."
        )
    elif not version:
        version = ver("nhssynth")

    run_type = config_dict.pop("run_type", None)

    if run_type == "pipeline":
        modules_to_run = PIPELINE
    else:
        modules_to_run = [x for x in config_dict.keys() | {run_type} if x in valid_run_types]
        if not args.custom_pipeline:
            modules_to_run = sorted(modules_to_run, key=lambda x: PIPELINE.index(x))

    if not modules_to_run:
        warnings.warn(
            "Missing or invalid `run_type` and / or module specification hierarchy in `config/{args.input_config}.yaml`, defaulting to a full run of the pipeline"
        )
        modules_to_run = PIPELINE

    # Get all possible default arguments by scraping the top level `parser` and the appropriate sub-parser for the `run_type`
    args_dict, required_args = get_default_and_required_args(
        parser, filter_dict(all_subparsers, modules_to_run, include=True)
    )

    # Find the non-default arguments amongst passed `args` by seeing which of them are different to the entries of `args_dict`
    non_default_passed_args_dict = {
        k: v
        for k, v in vars(args).items()
        if k in ["input_config", "custom_pipeline"] or (k in args_dict and k != "func" and v != args_dict[k])
    }

    # Overwrite the default arguments with the ones from the yaml file
    args_dict.update(flatten_dict(config_dict))

    # Overwrite the result of the above with any non-default CLI args
    args_dict.update(non_default_passed_args_dict)

    # Create a new Namespace using the assembled dictionary
    new_args = argparse.Namespace(**args_dict)
    assert getattr(
        new_args, "dataset"
    ), "No dataset specified in the passed config file, provide one with the `--dataset` argument or add it to the config file"
    assert all(
        getattr(new_args, req_arg["arg"]) for req_arg in required_args
    ), f"Required arguments are missing from the passed config file: {[ra['module'] + ':' + ra['arg'] for ra in required_args if not getattr(new_args, ra['arg'])]}"

    # Run the appropriate execution function(s)
    if not new_args.seed:
        warnings.warn("No seed has been specified, meaning the results of this run may not be reproducible.")
    new_args.version = version
    new_args.modules_to_run = modules_to_run
    new_args.module_handover = {}
    for module in new_args.modules_to_run:
        MODULE_MAP[module](new_args)

    return new_args


def get_modules_to_run(executor: Callable) -> list[str]:
    """
    Get the list of modules to run from the passed executor function.

    Args:
        executor: The executor function to run.

    Returns:
        A list of module names to run.
    """
    if executor == run_pipeline:
        return PIPELINE
    else:
        return [get_key_by_value({mn: mc.func for mn, mc in MODULE_MAP.items()}, executor)]


def assemble_config(
    args: argparse.Namespace,
    all_subparsers: dict[str, argparse.ArgumentParser],
) -> dict[str, Any]:
    """
    Assemble and arrange a module-wise nested configuration dictionary from parsed command-line arguments to be output as a YAML record.

    Args:
        args: A namespace object containing all parsed command-line arguments.
        all_subparsers: A dictionary mapping module names to subparser objects.

    Returns:
        A dictionary containing configuration information extracted from `args` in a module-wise nested format that is YAML-friendly.

    Raises:
        ValueError: If a module specified in `args.modules_to_run` is not in `all_subparsers`.
    """
    args_dict = vars(args)

    # Filter out the keys that are not relevant to the config file
    args_dict = filter_dict(
        args_dict, {"func", "experiment_name", "save_config", "save_config_path", "module_handover"}
    )
    for k in args_dict.copy().keys():
        # Remove empty metric lists from the config
        if "_metrics" in k and not args_dict[k]:
            args_dict.pop(k)

    modules_to_run = args_dict.pop("modules_to_run")
    if len(modules_to_run) == 1:
        run_type = modules_to_run[0]
    elif modules_to_run == PIPELINE:
        run_type = "pipeline"
    else:
        raise ValueError(f"Invalid value for `modules_to_run`: {modules_to_run}")

    # Generate a dictionary containing each module's name from the run, with all of its possible corresponding config args
    module_args = {
        module_name: [action.dest for action in all_subparsers[module_name]._actions if action.dest != "help"]
        for module_name in modules_to_run
    }

    # Use the flat namespace to populate a nested (by module) dictionary of config args and values
    out_dict = {}
    for module_name in modules_to_run:
        for k in args_dict.copy().keys():
            # We want to keep dataset, experiment_name, seed and save_config at the top-level as they are core args
            if k in module_args[module_name] and k not in {
                "version",
                "dataset",
                "experiment_name",
                "seed",
                "save_config",
            }:
                if module_name not in out_dict:
                    out_dict[module_name] = {}
                v = args_dict.pop(k)
                if v is not None:
                    out_dict[module_name][k] = v

    # Assemble the final dictionary in YAML-compliant form
    return {**({"run_type": run_type} if run_type else {}), **args_dict, **out_dict}


def write_config(
    args: argparse.Namespace,
    all_subparsers: dict[str, argparse.ArgumentParser],
) -> None:
    """
    Assembles a configuration dictionary from the run config and writes it to a YAML file at the location specified by `args.save_config_path`.

    Args:
        args: A namespace containing the run's configuration.
        all_subparsers: A dictionary containing all subparsers for the config args.
    """
    experiment_name = args.experiment_name
    args_dict = assemble_config(args, all_subparsers)
    with open(f"experiments/{experiment_name}/config_{experiment_name}.yaml", "w") as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False, sort_keys=False)
