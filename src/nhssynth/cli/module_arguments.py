import argparse


def add_all_module_args(parser: argparse.ArgumentParser):
    dataloader_group = parser.add_argument_group(title="dataloader")
    add_dataloader_args(dataloader_group)
    structure_group = parser.add_argument_group(title="structure")
    add_structure_args(structure_group)
    model_group = parser.add_argument_group(title="model")
    add_model_args(model_group)
    evaluation_group = parser.add_argument_group(title="evaluation")
    add_evaluation_args(evaluation_group)
    plotting_group = parser.add_argument_group(title="plotting")
    add_plotting_args(plotting_group)


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input-config",
        "-c",
        required=True,
        help="Specify the config file.",
    )
    overrides_group = parser.add_argument_group(title="overrides")
    # TODO is there a way to do this using `add_all_module_args`, i.e. can we nest groups? Doesn't seem to work
    add_dataloader_args(overrides_group, override=True)
    add_structure_args(overrides_group, override=True)
    add_model_args(overrides_group, override=True)
    add_evaluation_args(overrides_group, override=True)
    add_plotting_args(overrides_group, override=True)


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
