import argparse

from nhssynth.common.constants import *


def add_dataloader_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing dataloader module sub-parser instance."""
    dataloader_grp = parser.add_argument_group(title=group_title)
    dataloader_grp.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="the directory containing the chosen dataset",
    )
    dataloader_grp.add_argument(
        "--index-col",
        default=None,
        choices=[None, 0],
        help="indicate whether the csv file's 0th column is an index column, such that pandas can ignore it",
    )
    dataloader_grp.add_argument(
        "--sdv-workflow",
        action="store_true",
        help="utilise the SDV synthesizer workflow for transformation and metadata, rather than a `HyperTransformer` from RDT",
    )
    dataloader_grp.add_argument(
        "--allow-null-transformers",
        action="store_true",
        help="allow null / None transformers, i.e. leave some columns as they are",
    )
    dataloader_grp.add_argument(
        "--collapse-yaml",
        action="store_true",
        help="use aliases and anchors in the output metadata yaml, this will make it much more compact",
    )
    # TODO might be good to have something like this, needs some thought in how to only apply to appropriate transformers, without overriding metadata
    # dataloader_grp.add_argument(
    #     "--imputation-strategy",
    #     default=None,
    #     help="imputation strategy for missing values, pick one of None, `mean`, `mode`, or a number",
    # )
    dataloader_grp.add_argument(
        "--synthesizer",
        type=str,
        default="TVAE",
        choices=list(SDV_SYNTHESIZER_CHOICES.keys()),
        help="pick a synthesizer to use (note this can also be specified in the model module, these must match)",
    )


def add_structure_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    pass


def add_model_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing model module sub-parser instance."""
    model_grp = parser.add_argument_group(title=group_title)
    model_grp.add_argument(
        "--model-file",
        type=str,
        default="_model",
        help="specify the filename of the model to be saved in `experiments/<EXPERIMENT_NAME>/`, defaults to `<args.real_data>_model.pt`",
    )
    model_grp.add_argument(
        "--use-gpu",
        action="store_true",
        help="use the GPU for training",
    )
    model_grp.add_argument(
        "--non-private-training",
        action="store_true",
        help="train the model in a non-private way",
    )
    model_grp.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    model_grp.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of epochs to train for",
    )
    model_grp.add_argument(
        "--tracked-metrics",
        type=str,
        nargs="+",
        default=TRACKED_METRIC_CHOICES,
        help="metrics to track during training of the DPVAE model",
        choices=TRACKED_METRIC_CHOICES,
    )
    model_grp.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="the latent dimension of the model",
    )
    model_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="the hidden dimension of the model",
    )
    model_grp.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="the learning rate for the model",
    )
    model_grp.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="the batch size for the model",
    )
    model_grp.add_argument(
        "--patience",
        type=int,
        default=5,
        help="how many epochs the model is allowed to train for without improvement",
    )
    model_grp.add_argument(
        "--delta",
        type=int,
        default=10,
        help="the difference in successive ELBO values that register as an 'improvement'",
    )
    model_grp.add_argument(
        "--target-epsilon",
        type=float,
        default=1.0,
        help="the target epsilon for differential privacy",
    )
    model_grp.add_argument(
        "--target-delta",
        type=float,
        default=1e-5,
        help="the target delta for differential privacy",
    )
    model_grp.add_argument(
        "--max-grad-norm",
        type=int,
        default=10,
        help="the clipping threshold for gradients (only relevant under differential privacy)",
    )


def add_evaluation_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing evaluation module sub-parser instance."""
    evaluation_grp = parser.add_argument_group(title=group_title)
    # evaluation_grp.add_argument(
    #     "--prepared-data",
    #     type=str,
    #     default="_typed",
    #     help="specify the filename of the synthetic data to be read in `experiments/<EXPERIMENT_NAME>/`, defaults to `<args.real_data>_typed.pkl`",
    # )
    # evaluation_grp.add_argument(
    #     "--synthetic-data",
    #     type=str,
    #     default="_synthetic",
    #     help="specify the filename of the synthetic data to be read in `experiments/<EXPERIMENT_NAME>/`, defaults to `<args.real_data>_synthetic.pkl`",
    # )
    evaluation_grp.add_argument(
        "--detection-metrics",
        type=str,
        default=None,
        nargs="+",
        choices=list(SDV_DETECTION_METRIC_CHOICES.keys()),
    )


def add_plotting_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing plotting module sub-parser instance."""
    pass
