import argparse

from nhssynth.cli.config import get_modules_to_run, read_config, write_config
from nhssynth.cli.module_arguments import add_top_level_args
from nhssynth.cli.module_setup import MODULE_MAP, add_subparser


def run() -> None:
    """CLI for preparing, training and evaluating a synthetic data generator."""

    parser = argparse.ArgumentParser(
        prog="nhssynth", description="CLI for preparing, training and evaluating a synthetic data generator."
    )
    add_top_level_args(parser)

    # Below we instantiate one subparser for each module + one for running with a config file and one for
    # doing a full pipeline run with CLI-specified config
    subparsers = parser.add_subparsers()

    all_subparsers = {
        name: add_subparser(subparsers, name, option_config) for name, option_config in MODULE_MAP.items()
    }

    args = parser.parse_args()

    # Use get to return None when no function has been set, i.e. user made no running choice
    executor = vars(args).get("func")

    if executor:
        args.modules_to_run = get_modules_to_run(executor)
        executor(args)
    elif hasattr(args, "input_config"):
        args = read_config(args, parser, all_subparsers)
    else:
        parser.print_help()

    # Whenever either are specified, we want to dump the configuration to allow for this run to be replicated
    if args.save_config or args.save_config_path:
        write_config(args, all_subparsers)

    print("Complete!")
