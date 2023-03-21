import argparse

from nhssynth.cli.module_setup import *
from nhssynth.utils.constants import TIME


def add_module_subparser(subparsers: argparse._SubParsersAction, name: str):
    parser = subparsers.add_parser(
        name=name,
        description=MODULE_MAP[name].description,
        help=MODULE_MAP[name].help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    MODULE_MAP[name].args_func(parser)
    parser.set_defaults(func=MODULE_MAP[name].func)
    return parser


def add_top_level_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-rn",
        "--run-name",
        default=TIME,
        help=f"name the run to affect logging, config and outputs, defaults to current time, i.e. `{TIME}`",
    )
    parser.add_argument(
        "-sc", "--save-config", action="store_true", help="provide this flag to save the config provided via the cli"
    )
    parser.add_argument(
        "-scp",
        "--save-config-path",
        help="where to save the config when `-sc` is provided, defaults to `config/dump_<RUN_NAME>.yaml`",
    )
    parser.add_argument("-s", "--seed", help="specify a seed for reproducibility")
