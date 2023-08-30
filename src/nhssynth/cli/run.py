import argparse
import time
import warnings

from nhssynth.cli.config import get_modules_to_run, read_config, write_config
from nhssynth.cli.module_setup import MODULE_MAP, add_subparser
from nhssynth.common.strings import format_timedelta


def run(sysargv) -> None:
    print("\nStarting up the NHSSynth CLI! 🚀\n")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog="nhssynth",
        description="CLI for preparing, training and evaluating a synthetic data generator.",
    )

    # Below we instantiate one subparser for each module + one for running with a config file and one for
    # doing a full pipeline run with CLI-specified config
    subparsers = parser.add_subparsers()
    all_subparsers = {
        name: add_subparser(subparsers, name, option_config) for name, option_config in MODULE_MAP.items()
    }

    args = parser.parse_args(sysargv)

    executor = vars(args).get("func", None)
    if executor:
        if hasattr(args, "seed") and not args.seed:
            warnings.warn("No seed has been specified, meaning the results of this run may not be reproducible.")
        args.modules_to_run = get_modules_to_run(executor)
        args.module_handover = {}
        executor(args)
    elif hasattr(args, "input_config"):
        args = read_config(args, parser, all_subparsers)
    else:
        return parser.print_help()

    if args.save_config:
        write_config(args, all_subparsers)

    print(f"Finished! 🎉 (Total run time: {format_timedelta(start_time, time.time())})")
