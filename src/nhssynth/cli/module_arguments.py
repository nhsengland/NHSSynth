"""Define arguments for each of the modules' CLI sub-parsers."""
import argparse

from nhssynth.cli.model_arguments import add_model_specific_args
from nhssynth.common.constants import *
from nhssynth.modules.dataloader.metadata import MISSINGNESS_STRATEGIES
from nhssynth.modules.model import MODELS


class AllChoicesDefault(argparse.Action):
    """
    Customised argparse action for defaulting to the full list of choices if only the argument's flag is supplied:
    (i.e. user passes `--metrics` with no follow up list of metric groups => all metric groups will be executed).

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
        "--constraint-graph",
        type=str,
        default="_constraint_graph",
        help="the name of the html file to write the constraint graph to, defaults to `<DATASET>_constraint_graph`",
    )
    group.add_argument(
        "--collapse-yaml",
        action="store_true",
        help="use aliases and anchors in the output metadata yaml, this will make it much more compact",
    )
    group.add_argument(
        "--missingness",
        type=str,
        default="augment",
        choices=MISSINGNESS_STRATEGIES,
        help="how to handle missing values in the dataset",
    )
    group.add_argument(
        "--impute",
        type=str,
        default=None,
        help="the imputation strategy to use, ONLY USED if <MISSINGNESS> is set to 'impute', choose from: 'mean', 'median', 'mode', or any specific value (e.g. '0')",
    )
    group.add_argument(
        "--write-csv",
        action="store_true",
        help="write the transformed real data to a csv file",
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
        choices=MODELS,
        help="the model architecture(s) to train",
    )
    group.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how many times to repeat the training process per model architecture (<SEED> is incremented each time)",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=32,
        help="the batch size for the model",
    )
    group.add_argument(
        "--num-epochs",
        type=int,
        nargs="+",
        default=100,
        help="number of epochs to train for",
    )
    group.add_argument(
        "--patience",
        type=int,
        nargs="+",
        default=5,
        help="how many epochs the model is allowed to train for without improvement",
    )
    group.add_argument(
        "--displayed-metrics",
        type=str,
        nargs="+",
        default=TRACKED_METRICS,
        help="metrics to display during training of the model",
        choices=TRACKED_METRICS,
    )
    group.add_argument(
        "--use-gpu",
        action="store_true",
        help="use the GPU for training",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="the number of samples to generate from the model, defaults to the size of the original dataset",
    )
    privacy_group = parser.add_argument_group(title="model privacy options")
    privacy_group.add_argument(
        "--target-epsilon",
        type=float,
        nargs="+",
        default=1.0,
        help="the target epsilon for differential privacy",
    )
    privacy_group.add_argument(
        "--target-delta",
        type=float,
        nargs="+",
        help="the target delta for differential privacy, defaults to `1 / len(dataset)` if not specified",
    )
    privacy_group.add_argument(
        "--max-grad-norm",
        type=float,
        nargs="+",
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
        const=list(METRIC_CHOICES[name].keys()),
        choices=list(METRIC_CHOICES[name].keys()),
        help=f"run the {name.lower()} evaluation",
    )


def add_evaluation_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing evaluation module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--downstream-tasks",
        "--tasks",
        action="store_true",
        help="run the downstream tasks evaluation",
    )
    group.add_argument(
        "--tasks-dir",
        type=str,
        default="./tasks",
        help="the directory containing the downstream tasks to run, this directory must contain a folder called <DATASET> containing the tasks to run",
    )
    group.add_argument(
        "--aequitas",
        action="store_true",
        help="run the aequitas fairness evaluation (note this runs for each of the downstream tasks)",
    )
    group.add_argument(
        "--aequitas-attributes",
        type=str,
        nargs="+",
        default=None,
        help="the attributes to use for the aequitas fairness evaluation, defaults to all attributes",
    )
    group.add_argument(
        "--key-numerical-fields",
        type=str,
        nargs="+",
        default=None,
        help="the numerical key field attributes to use for SDV privacy evaluations",
    )
    group.add_argument(
        "--sensitive-numerical-fields",
        type=str,
        nargs="+",
        default=None,
        help="the numerical sensitive field attributes to use for SDV privacy evaluations",
    )
    group.add_argument(
        "--key-categorical-fields",
        type=str,
        nargs="+",
        default=None,
        help="the categorical key field attributes to use for SDV privacy evaluations",
    )
    group.add_argument(
        "--sensitive-categorical-fields",
        type=str,
        nargs="+",
        default=None,
        help="the categorical sensitive field attributes to use for SDV privacy evaluations",
    )
    for name in METRIC_CHOICES:
        generate_evaluation_arg(group, name)


def add_plotting_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False) -> None:
    """Adds arguments to an existing plotting module sub-parser instance."""
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--plot-quality",
        action="store_true",
        help="plot the SDV quality report",
    )
    group.add_argument(
        "--plot-diagnostic",
        action="store_true",
        help="plot the SDV diagnostic report",
    )
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


def add_dashboard_args(parser: argparse.ArgumentParser, group_title: str, overrides: bool = False):
    group = parser.add_argument_group(title=group_title)
    group.add_argument(
        "--file-size-limit",
        type=str,
        default="1000",
        help="the maximum file size to upload in MB",
    )
    group.add_argument(
        "--dont-load",
        action="store_true",
        help="don't attempt to automatically load data into the dashboard",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="print all output from the dashboard",
    )
