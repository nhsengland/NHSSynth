import argparse
from typing import Final

from nhssynth.common.constants import *


def get_dataset_parser(overrides=False) -> argparse.ArgumentParser:
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
        help="specify a seed for reproducibility",
    )
    core_grp.add_argument(
        "--save-config",
        action="store_true",
        help="save the config provided via the cli",
    )
    return core


COMMON_TITLE: Final = "starting any of the following args with `_` defaults to a suffix on DATASET (e.g. `_metadata` -> `DATASET_metadata`);\nall filenames are relative to `experiments/<EXPERIMENT_NAME>/` unless otherwise stated"


def get_metadata_parser(overrides=False) -> argparse.ArgumentParser:
    metadata = argparse.ArgumentParser(add_help=False)
    metadata_grp = metadata.add_argument_group(title=COMMON_TITLE)
    metadata_grp.add_argument(
        "-m",
        "--metadata",
        type=str,
        default="_metadata",
        help="filename of the metadata, NOTE that `dataloader` attempts to read this from `<DATA_DIR>`",
    )
    return metadata


def get_typed_parser(overrides=False) -> argparse.ArgumentParser:
    typed = argparse.ArgumentParser(add_help=False)
    typed_grp = typed.add_argument_group(title=COMMON_TITLE)
    typed_grp.add_argument(
        "--typed",
        type=str,
        default="_typed",
        help="filename of the typed data",
    )
    return typed


def get_prepared_parser(overrides=False) -> argparse.ArgumentParser:
    prepared = argparse.ArgumentParser(add_help=False)
    prepared_grp = prepared.add_argument_group(title=COMMON_TITLE)
    prepared_grp.add_argument(
        "--prepared",
        type=str,
        default="_prepared",
        help="filename of the prepared data",
    )
    prepared_grp.add_argument(
        "--metatransformer",
        type=str,
        default="_metatransformer",
        help="filename of the `MetaTransformer` used to prepare the data",
    )
    return prepared


def get_synthetic_parser(overrides=False) -> argparse.ArgumentParser:
    synthetic = argparse.ArgumentParser(add_help=False)
    synthetic_grp = synthetic.add_argument_group(title=COMMON_TITLE)
    synthetic_grp.add_argument(
        "--synthetic",
        type=str,
        default="_synthetic",
        help="filename of the synthetic data",
    )
    return synthetic


COMMON_PARSERS: Final = {
    "dataset": get_dataset_parser,
    "core": get_core_parser,
    "metadata": get_metadata_parser,
    "typed": get_typed_parser,
    "prepared": get_prepared_parser,
    "synthetic": get_synthetic_parser,
}
