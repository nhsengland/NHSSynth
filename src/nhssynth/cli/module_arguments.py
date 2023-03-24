import argparse

from nhssynth.utils.constants import TIME


def add_top_level_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-rn",
        "--run-name",
        default=TIME,
        help=f"name the run to affect logging, config and outputs, defaults to current time, i.e. `{TIME}`",
    )
    parser.add_argument("-sc", "--save-config", action="store_true", help="save the config provided via the cli")
    parser.add_argument(
        "-scp",
        "--save-config-path",
        help="where to save the config when `-sc` is provided, defaults to `experiments/<RUN_NAME>/config_<RUN_NAME>.yaml`",
    )
    parser.add_argument("-s", "--seed", help="specify a seed for reproducibility")


def add_dataloader_args(parser: argparse.ArgumentParser, override=False):
    parser.add_argument(
        "-i",
        "--input-file",
        required=(not override),
        help="the name of the `.csv` file to prepare",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="_prepared",
        help="where to write the prepared data, defaults to `experiments/<args.run_name>/<args.input_file>_prepared.csv`",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default="./data",
        help="the directory to read and write data from and to",
    )
    parser.add_argument(
        "-m",
        "--metadata-file",
        default="_metadata",
        help="metadata for the input data, defaults to `<args.dir>/<args.input_file>_metadata.yaml`",
    )
    parser.add_argument(
        "-ic",
        "--index-col",
        default=None,
        choices=[None, 0],
        help="indicate whether the csv file's 0th column is an index column, such that pandas can ignore it",
    )
    parser.add_argument(
        "-sdv",
        "--sdv-workflow",
        action="store_true",
        help="utilise the SDV synthesizer workflow for transformation and metadata, rather than a `HyperTransformer` from RDT",
    )
    parser.add_argument(
        "-ant",
        "--allow-null-transformers",
        action="store_true",
        help="allow null / None transformers, i.e. leave some columns as they are",
    )
    parser.add_argument(
        "-cy",
        "--collapse-yaml",
        action="store_true",
        help="use aliases and anchors in the output metadata yaml, this will make it much more compact",
    )
    parser.add_argument(
        "-is",
        "--imputation-strategy",
        default="mean",
        choices=["mean", "median", "cull"],
        help="imputation strategy for missing values",
    )


def add_structure_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_model_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_evaluation_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_plotting_args(parser: argparse.ArgumentParser, override=False):
    pass
