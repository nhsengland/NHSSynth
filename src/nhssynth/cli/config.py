import argparse
import warnings
from typing import Any

import nhssynth.cli.module_setup as ms
import yaml
from nhssynth.utils import flatten_dict, get_key_by_value


def get_default_and_required_args(
    top_parser: argparse.ArgumentParser,
    module_parsers: list[argparse.ArgumentParser],
) -> tuple[dict[str, Any], list[str]]:
    """
    Get the default and required arguments for the top-level parser and the current run's corresponding list of module parsers.

    Args:
        top_parser: The top-level parser.
        module_parsers: The list of module-level parsers.

    Returns:
        A tuple containing two elements:
            - A dictionary containing all arguments and their default values.
            - A list of the names of the required arguments.
    """
    all_actions = top_parser._actions + [action for sub_parser in module_parsers for action in sub_parser._actions]
    defaults = {}
    required_args = []
    for action in all_actions:
        if action.dest not in ["help", "==SUPPRESS=="]:
            defaults[action.dest] = action.default
            if action.required:
                required_args.append(action.dest)
    return defaults, required_args


def read_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    all_subparsers: dict[str, argparse.ArgumentParser],
) -> argparse.Namespace:
    """
    Hierarchically assembles a config Namespace object for the inferred modules to run and executes.

    1. Load the YAML file containing the config to read from
    2. Check a valid `run_type` is specified or infer it and determine the list of `modules_to_run`
    3. Establish the appropriate default config from the parser and `all_subparsers` for the `modules_to_run`
    4. Overwrite this config with the specified subset (or full set) of config in the YAML file
    5. Overwrite again with passed command-line `args` (these are considered 'overrides')
    6. Run the appropriate module(s) or pipeline with the resulting config

    Args:
        args: Namespace object containing arguments from the command line
        parser: top-level ArgumentParser object
        all_subparsers: dictionary of ArgumentParser objects, one for each module

    Returns:
        Namespace object containing the assembled configuration settings

    Raises:
        AssertionError: if any required arguments are missing from the configuration file
    """
    # Open the passed yaml file and load into a dictionary
    with open(f"config/{args.input_config}.yaml") as stream:
        config_dict = yaml.safe_load(stream)

    valid_run_types = [x for x in all_subparsers.keys() if x != "config"]

    run_type = config_dict.pop("run_type", None)
    # TODO Check this covers all bases
    if run_type == "pipeline":
        modules_to_run = ms.PIPELINE
    else:
        modules_to_run = [x for x in config_dict.keys() | {run_type} if x in valid_run_types]
        if not args.custom_pipeline:
            modules_to_run = sorted(modules_to_run, key=lambda x: ms.PIPELINE.index(x))

    if not modules_to_run:
        warnings.warn(
            "Missing or invalid `run_type` and / or module specification hierarchy in `config/{args.input_config}.yaml`, defaulting to a full run of the pipeline"
        )
        modules_to_run = ms.PIPELINE

    # Get all possible default arguments by scraping the top level `parser` and the appropriate sub-parser for the `run_type`
    args_dict, required_args = get_default_and_required_args(
        parser, [all_subparsers[module_name] for module_name in modules_to_run]
    )

    # Find the non-default arguments amongst passed `args` by seeing which of them are different to the entries of `args_dict`
    non_default_passed_args_dict = {
        k: v
        for k, v in vars(args).items()
        if k in ["input_config", "custom_pipeline"] or (k != "func" and v != args_dict[k])
    }

    # Overwrite the default arguments with the ones from the yaml file
    args_dict.update(flatten_dict(config_dict))

    # Overwrite the result of the above with any non-default CLI args
    args_dict.update(non_default_passed_args_dict)

    # Create a new Namespace using the assembled dictionary
    new_args = argparse.Namespace(**args_dict)
    assert all(
        getattr(new_args, req_arg) for req_arg in required_args
    ), "Required arguments are missing from the passed config file"

    # Run the appropriate execution function(s)
    for module in modules_to_run:
        ms.MODULE_MAP[module].func(new_args)

    new_args.modules_to_run = modules_to_run
    return new_args


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
    modules_to_run = args_dict.pop("modules_to_run", None)
    if not modules_to_run:
        run_type = get_key_by_value({mn: mc.func for mn, mc in ms.MODULE_MAP.items()}, args_dict["func"])
        modules_to_run = ms.PIPELINE if run_type == "pipeline" else run_type
    elif len(modules_to_run) == 1:
        run_type = modules_to_run[0]
    elif modules_to_run == ms.PIPELINE:
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
            if k in module_args[module_name]:
                if out_dict.get(module_name):
                    out_dict[module_name].update({k: args_dict.pop(k)})
                else:
                    out_dict[module_name] = {k: args_dict.pop(k)}

    # Assemble the final dictionary in YAML-compliant form
    return {
        **({"run_type": run_type} if run_type else {}),
        **{k: v for k, v in args_dict.items() if k not in {"func", "run_name", "save_config", "save_config_path"}},
        **out_dict,
    }


def write_config(
    args: argparse.Namespace,
    all_subparsers: dict[str, argparse.ArgumentParser],
) -> None:
    """
    Assembles a configuration dictionary from the run config and writes it to a YAML file at the location specified by `args.save_config_path`.

    Args:
        args: A namespace containing the run's configuration.
        all_subparsers: A dictionary containing all subparsers for the config args.

    Returns:
        None.
    """
    args_dict = assemble_config(args, all_subparsers)
    with open(f"{args.save_config_path}", "w") as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False, sort_keys=False)
