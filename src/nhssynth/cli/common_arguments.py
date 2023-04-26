"""
Functions to define the CLI's "common" arguments, i.e. those that can be applied to either:
 - All module argument lists, e.g. --dataset, --seed, etc.
 - A subset of module argument lists, e.g. --synthetic, --typed, etc.
"""
import argparse
from typing import Final

from nhssynth.common.constants import *


def get_dataset_parser(overrides=False) -> argparse.ArgumentParser:
    """Create a common parser for specifying the dataset"""
    dataset = argparse.ArgumentParser(add_help=False)
    dataset_grp = dataset.add_argument_group(title="options")
    dataset_grp.add_argument(
        "-d",
        "--dataset",
        required=(not overrides),
        type=str,
        help="the name of the dataset to experiment with, should be present in `<DATA_DIR>`",
    )
    return dataset


def get_core_parser(overrides=False) -> argparse.ArgumentParser:
    """Create a common parser for specifying the core args (except for dataset which is separate)"""
    core = argparse.ArgumentParser(add_help=False)
    core_grp = core.add_argument_group(title="options")
    core_grp.add_argument(
        "-e",
        "--experiment-name",
        type=str,
        default=TIME,
        help=f"name the experiment run to affect logging, config, and default-behaviour io",
    )
    core_grp.add_argument(
        "-s",
        "--seed",
        type=int,
        help="specify a seed for reproducibility, this is a recommended option for reproducibility",
    )
    core_grp.add_argument(
        "--save-config",
        action="store_true",
        help="save the config provided via the cli, this is a recommended option for reproducibility",
    )
    return core


COMMON_TITLE: Final = "starting any of the following args with `_` defaults to a suffix on DATASET (e.g. `_metadata` -> `<DATASET>_metadata`);\nall filenames are relative to `experiments/<EXPERIMENT_NAME>/` unless otherwise stated"


def suffix_parser_generator(name: str, help: str, required: bool = False) -> argparse.ArgumentParser:
    """Generator function for creating common parsers for specifying a potential suffix filename

    Args:
        name: the name of the argument
        help: the help message for the argument
        required: whether the argument is required
    """

    def get_parser(overrides: bool = False) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser_grp = parser.add_argument_group(title=COMMON_TITLE)
        parser_grp.add_argument(
            f"--{name}",
            required=required and not overrides,
            type=str,
            default=f"_{name}",
            help=help,
        )
        return parser

    return get_parser


COMMON_PARSERS: Final = {
    "dataset": get_dataset_parser,
    "core": get_core_parser,
    "metadata": suffix_parser_generator(
        "metadata", "filename of the metadata, NOTE that `dataloader` attempts to read this from `<DATA_DIR>`"
    ),
    "typed": suffix_parser_generator("typed", "filename of the typed data"),
    "prepared": suffix_parser_generator("prepared", "filename of the prepared data"),
    "metatransformer": suffix_parser_generator(
        "metatransformer", "filename of the `MetaTransformer` used to prepare the data"
    ),
    "synthetic": suffix_parser_generator("synthetic", "filename of the synthetic data"),
    "report": suffix_parser_generator("report", "filename of the (collection of) report(s)"),
}
