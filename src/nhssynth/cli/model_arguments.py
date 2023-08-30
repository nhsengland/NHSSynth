"""Define arguments for each of the model classes."""
import argparse

from nhssynth.common.constants import *


def add_model_specific_args(group: argparse._ArgumentGroup, name: str, overrides: bool = False) -> None:
    """Adds arguments to an existing group according to `name`."""
    if name == "VAE":
        add_vae_args(group, overrides)


def add_vae_args(group: argparse._ArgumentGroup, overrides: bool = False) -> None:
    """Adds arguments to an existing group for the VAE model."""
    group.add_argument(
        "--encoder-latent-dim",
        type=int,
        nargs="+",
        help="the latent dimension of the encoder",
    )
    group.add_argument(
        "--encoder-hidden-dim",
        type=int,
        nargs="+",
        help="the hidden dimension of the encoder",
    )
    group.add_argument(
        "--encoder-activation",
        type=str,
        nargs="+",
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the encoder",
    )
    group.add_argument(
        "--encoder-learning-rate",
        type=float,
        nargs="+",
        help="the learning rate for the encoder",
    )
    group.add_argument(
        "--decoder-latent-dim",
        type=int,
        nargs="+",
        help="the latent dimension of the decoder",
    )
    group.add_argument(
        "--decoder-hidden-dim",
        type=int,
        nargs="+",
        help="the hidden dimension of the decoder",
    )
    group.add_argument(
        "--decoder-activation",
        type=str,
        nargs="+",
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the decoder",
    )
    group.add_argument(
        "--decoder-learning-rate",
        type=float,
        nargs="+",
        help="the learning rate for the decoder",
    )
    group.add_argument(
        "--shared-optimizer",
        action="store_true",
        help="whether to use a shared optimizer for the encoder and decoder",
    )
