"""Define arguments for each of the modules' CLI sub-parsers."""
import argparse

from nhssynth.cli.model_arguments import add_model_specific_args
from nhssynth.common.constants import *
from nhssynth.modules.model import MODELS


class AllChoicesDefault(argparse.Action):
    """
    Customized argparse action for defaulting to the full list of choices if only the flag is supplied.

    Notes:
        1) If no `option_string` is supplied: set to default value (`self.default`)
        2) If `option_string` is supplied:
            a) If `values` are supplied, set to list of values
            b) If no `values` are supplied, set to `self.const`, if `self.const` is not set, set to `self.default`
    """

    def __call__(self, parser, namespace, values=None, option_string=None):
        if values:
            setattr(namespace, self.dest, values)
        elif option_string:
            setattr(namespace, self.dest, self.const if self.const else self.default)
        else:
            setattr(namespace, self.dest, self.default)


def add_dataloader_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing dataloader module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="the directory containing the chosen dataset",
    )
    group.add_argument(
        "--index-col",
        default=None,
        choices=[None, 0],
        help="indicate whether the csv file's 0th column is an index column, such that pandas can ignore it",
    )
    group.add_argument(
        "--allow-null-transformers",
        action="store_true",
        help="allow null / None transformers, i.e. leave some columns as they are",
    )
    group.add_argument(
        "--collapse-yaml",
        action="store_true",
        help="use aliases and anchors in the output metadata yaml, this will make it much more compact",
    )
    group.add_argument(
        "--synthesizer",
        type=str,
        default="TVAE",
        choices=list(SDV_SYNTHESIZERS.keys()),
        help="pick a synthesizer to use (note this can also be specified in the model module, these must match)",
    )


def add_structure_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    pass


def add_model_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing model module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--architecture",
        type=str,
        nargs="+",
        default=["VAE"],
        choices=list(MODELS.keys()),
        help="the model architecture(s) to train",
    )
    group.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how many times to repeat the training process per model architecture (<SEED> is incremented each time)",
    )
    group.add_argument(
        "--model-file",
        type=str,
        default="_model",
        help="specify the filename of the model to be saved in `experiments/<EXPERIMENT_NAME>/`, defaults to `<DATASET>_model.pt`",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="the batch size for the model",
    )
    group.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of epochs to train for",
    )
    group.add_argument(
        "--patience",
        type=int,
        default=5,
        help="how many epochs the model is allowed to train for without improvement",
    )
    group.add_argument(
        "--tracked-metrics",
        type=str,
        nargs="+",
        default=TRACKED_METRICS,
        help="metrics to track during training of the model",
        choices=TRACKED_METRICS,
    )
    group.add_argument(
        "--use-gpu",
        action="store_true",
        help="use the GPU for training",
    )
    privacy_group = parser.add_argument_group(title="model privacy options")
    privacy_group.add_argument(
        "--non-private",
        action="store_true",
        help="train the model in a non-private way",
    )
    privacy_group.add_argument(
        "--target-epsilon",
        type=float,
        default=1.0,
        help="the target epsilon for differential privacy",
    )
    privacy_group.add_argument(
        "--target-delta",
        type=float,
        help="the target delta for differential privacy, defaults to `1 / len(dataset)` if not specified",
    )
    privacy_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="the clipping threshold for gradients (only relevant under differential privacy)",
    )
    privacy_group.add_argument(
        "--secure-mode",
        action="store_true",
        help="Enable secure RNG via the `csprng` package to make privacy guarantees more robust, comes at a cost of performance and reproducibility",
    )
    for model_name in MODELS.keys():
        model_group = parser.add_argument_group(title=f"{model_name}-specific options")
        add_model_specific_args(model_group, model_name, overrides=overrides)


def generate_evaluation_arg(group, name):
    group.add_argument(
        f"--{'-'.join(name.split()).lower()}-metrics",
        type=str,
        default=None,
        nargs="*",
        action=AllChoicesDefault,
        const=list(SDV_METRICS[name].keys()),
        choices=list(SDV_METRICS[name].keys()),
        help=f"run the {name.lower()} evaluation",
    )


def add_evaluation_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing evaluation module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--diagnostic",
        action="store_true",
        help="run the diagnostic evaluation",
    )
    group.add_argument(
        "--quality",
        action="store_true",
        help="run the quality evaluation",
    )
    for name in SDV_METRICS:
        generate_evaluation_arg(group, name)


def add_plotting_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing plotting module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--plot-sdv-report",
        action="store_true",
        help="plot the SDV report",
    )
    group.add_argument(
        "--plot-tsne",
        action="store_true",
        help="plot the t-SNE embeddings of the real and synthetic data",
    )
