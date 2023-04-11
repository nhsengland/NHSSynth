import argparse

from nhssynth.common.constants import SDV_SYNTHESIZER_CHOICES, TIME


def add_top_level_args(parser: argparse.ArgumentParser) -> None:
    """Adds top-level arguments to an existing ArgumentParser instance."""
    parser.add_argument(
        "-e",
        "--experiment-name",
        type=str,
        default=TIME,
        help=f"name the experiment run to affect logging, config, and default-behaviour io, defaults to current time, i.e. `{TIME}`",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="save the config provided via the cli",
    )
    parser.add_argument(
        "--save-config-path",
        type=str,
        help="where to save the config when `-sc` is provided, defaults to `experiments/<experiment_name>/config_<experiment_name>.yaml`",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="specify a seed for reproducibility",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="write all outputs to disk, including those that are not strictly necessary i.e. intermediary outputs in a full pipeline run",
    )


def add_dataloader_args(parser: argparse.ArgumentParser, override=False) -> None:
    """Adds arguments to an existing dataloader module sub-parser instance."""
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=(not override),
        help="the name of the `.csv` file to prepare",
    )
    parser.add_argument(
        "--output-prepared",
        type=str,
        default="_prepared",
        help="where to write the prepared data, defaults to `experiments/<args.experiment_name>/<args.input_file>_prepared\{.csv/.pkl\}`, only used when `--write-all` is provided and/or this is a full pipeline run / one that involves the `model` module",
    )
    parser.add_argument(
        "--output-typed",
        type=str,
        default="_typed",
        help="where to write the prepared data, defaults to `experiments/<args.experiment_name>/<args.input_file>_typed.pkl`, only used when `--write-all` is provided and/or this is a full pipeline run / one that involves the `model` module",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="_metadata",
        help="metadata for the input data, defaults to `<args.input_dir>/<args.input_file>_metadata.yaml`",
    )
    parser.add_argument(
        "--discard-metadata",
        action="store_true",
        help="discard the generated metadata file (not recommended, this is required for reproducibility)",
    )
    parser.add_argument(
        "--metatransformer",
        type=str,
        default="_metatransformer",
        help="name of the file to dump the `metatransformer` object used on the input data, defaults to `experiments/<args.experiment_name>/<args.input_file>_metatransformer.pkl`, only used when `--write-all` is provided and/or this is a full pipeline run / one that involves the `model` module",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./data",
        help="the directory to read and write data from and to",
    )
    parser.add_argument(
        "--index-col",
        default=None,
        choices=[None, 0],
        help="indicate whether the csv file's 0th column is an index column, such that pandas can ignore it",
    )
    parser.add_argument(
        "--sdv-workflow",
        action="store_true",
        help="utilise the SDV synthesizer workflow for transformation and metadata, rather than a `HyperTransformer` from RDT",
    )
    parser.add_argument(
        "--allow-null-transformers",
        action="store_true",
        help="allow null / None transformers, i.e. leave some columns as they are",
    )
    parser.add_argument(
        "--collapse-yaml",
        action="store_true",
        help="use aliases and anchors in the output metadata yaml, this will make it much more compact",
    )
    # TODO might be good to have something like this, needs some thought in how to only apply to appropriate transformers, without overriding metadata
    # parser.add_argument(
    #     "--imputation-strategy",
    #     default=None,
    #     help="imputation strategy for missing values, pick one of None, `mean`, `mode`, or a number",
    # )
    parser.add_argument(
        "-s",
        "--synthesizer",
        type=str,
        default="TVAE",
        choices=list(SDV_SYNTHESIZER_CHOICES.keys()),
        help="pick a synthesizer to use (note this can also be specified in the model module, these must match)",
    )


def add_structure_args(parser: argparse.ArgumentParser, override=False) -> None:
    pass


def add_model_args(parser: argparse.ArgumentParser, override=False) -> None:
    """Adds arguments to an existing model module sub-parser instance."""
    parser.add_argument(
        "-r",
        "--real-data",
        type=str,
        help="name of the dataset, only REQUIRED when this module is run on its own",
    )
    parser.add_argument(
        "-p",
        "--prepared-data",
        type=str,
        default="_prepared",
        help="name of the prepared dataset to load from `experiments/<args.experiment_name>/`, defaults to `<args.real_data>_prepared.pkl`, only REQUIRED when this module is run on its own",
    )
    parser.add_argument(
        "-m",
        "--real-metatransformer",
        default="_metatransformer",
        type=str,
        help="name of the `.pkl` file of the MetaTransformer used on the prepared data to load from `experiments/<args.experiment_name>/`, defaults to `<args.real_data>_metatransformer.pkl` only REQUIRED when this module is run on its own",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="_model",
        help="specify the filename of the model to be saved in `experiments/<args.experiment_name>/`, defaults to `<args.real_data>_model.pt`",
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        default="_synthetic",
        help="specify the filename of the synthetic data to be written in `experiments/<args.experiment_name>/`, defaults to `<args.real_data>_synthetic.csv`",
    )
    parser.add_argument(
        "--discard-synthetic",
        action="store_true",
        help="do not output the synthetic data generated during the run",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="use the GPU for training",
    )
    parser.add_argument(
        "--non-private-training",
        action="store_true",
        help="train the model in a non-private way",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="the latent dimension of the model",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="the hidden dimension of the model",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="the learning rate for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="the batch size for the model",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="how many epochs the model is allowed to train for without improvement",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="the difference in successive ELBO values that register as an 'improvement'",
    )
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=1.0,
        help="the target epsilon for differential privacy",
    )
    parser.add_argument(
        "--target-delta",
        type=float,
        default=1e-5,
        help="the target delta for differential privacy",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=int,
        default=10,
        help="the clipping threshold for gradients (only relevant under differential privacy)",
    )


def add_evaluation_args(parser: argparse.ArgumentParser, override=False) -> None:
    """Adds arguments to an existing evaluation module sub-parser instance."""
    pass


def add_plotting_args(parser: argparse.ArgumentParser, override=False) -> None:
    """Adds arguments to an existing plotting module sub-parser instance."""
    pass
