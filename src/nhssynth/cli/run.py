import argparse

from nhssynth.cli.argparse import *
from nhssynth.cli.config import *


def run():

    parser = argparse.ArgumentParser(
        prog="SynthVAE", description="CLI for preparing, training and evaluating a synthetic data generator."
    )
    add_top_level_args(parser)

    # Below we instantiate one subparser for each module
    # + one for running with a config file and one for doing a full run with CLI-specified config
    subparsers = parser.add_subparsers()

    add_module_subparser(subparsers, "pipeline")
    config_parser = add_module_subparser(subparsers, "config")
    add_module_subparser(subparsers, "prepare")
    add_module_subparser(subparsers, "structure")
    add_module_subparser(subparsers, "train")
    add_module_subparser(subparsers, "evaluate")
    add_module_subparser(subparsers, "plot")

    args = parser.parse_args()

    # Use get to return None when no function has been set, i.e. user made no running choice
    executor = vars(args).get("func")

    # If `config` is the specified running choice, we mutate `args` using `parser` and `config_parser` in `read_config`
    # else we execute according to the user's choice
    # else we return `--help` if no choice has been passed, i.e. executor is None
    if executor == config:
        args = read_config(args, parser, config_parser)
    elif executor:
        executor(args)
    else:
        parser.parse_args(["--help"])

    # Whenever either are specified, we want to dump the configuration to allow for this run to be replicated
    if args.save_config or args.save_config_path:
        if not args.save_config_path:
            args.save_config_path = f"config/dump_{args.run_name}.yaml"
        write_config(args)

    print("Complete!")


if __name__ == "__main__":
    run()
