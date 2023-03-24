import argparse

from nhssynth.cli.config import *
from nhssynth.cli.module_arguments import add_top_level_args
from nhssynth.cli.module_setup import MODULE_MAP, add_subparser


def run():

    parser = argparse.ArgumentParser(
        prog="SynthVAE", description="CLI for preparing, training and evaluating a synthetic data generator."
    )
    add_top_level_args(parser)

    # Below we instantiate one subparser for each module + one for running with a config file and one for
    # doing a full pipeline run with CLI-specified config
    subparsers = parser.add_subparsers()

    # TODO can probably do this better as we dont actually need the `pipeline` or `config` subparsers in this dict
    all_subparsers = {
        name: add_subparser(subparsers, name, option_config) for name, option_config in MODULE_MAP.items()
    }

    args = parser.parse_args()

    # Use get to return None when no function has been set, i.e. user made no running choice
    executor = vars(args).get("func")

    # If `config` is the specified running choice, we mutate `args` in `read_config`
    # else we execute according to the user's choice
    # else we return `--help` if no choice has been passed, i.e. executor is None
    if not executor:
        args = read_config(args, parser, all_subparsers)
    elif executor:
        executor(args)
    else:
        parser.parse_args(["--help"])

    # Whenever either are specified, we want to dump the configuration to allow for this run to be replicated
    if args.save_config or args.save_config_path:
        if not args.save_config_path:
            args.save_config_path = f"experiments/{args.run_name}/config_{args.run_name}.yaml"
        write_config(args, all_subparsers)

    print("Complete!")


if __name__ == "__main__":
    run()
