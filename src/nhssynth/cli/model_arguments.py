"""Define arguments for each of the model classes."""

import argparse

from nhssynth.common.constants import ACTIVATION_FUNCTIONS


def add_model_specific_args(group: argparse._ArgumentGroup, name: str, overrides: bool = False) -> None:
    """Adds arguments to an existing group according to `name`."""
    if name == "VAE":
        add_vae_args(group, overrides)
    elif name == "GAN":
        add_gan_args(group, overrides)
    elif name == "TabularGAN":
        add_tabular_gan_args(group, overrides)


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


def add_gan_args(group: argparse._ArgumentGroup, overrides: bool = False) -> None:
    """Adds arguments to an existing group for the GAN model."""
    group.add_argument(
        "--n-units-conditional",
        type=int,
        help="the number of units in the conditional layer",
    )
    group.add_argument(
        "--generator-n-layers-hidden",
        type=int,
        help="the number of hidden layers in the generator",
    )
    group.add_argument(
        "--generator-n-units-hidden",
        type=int,
        help="the number of units in each hidden layer of the generator",
    )
    group.add_argument(
        "--generator-activation",
        type=str,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the generator",
    )
    group.add_argument(
        "--generator-batch-norm",
        action="store_true",
        help="whether to use batch norm in the generator",
    )
    group.add_argument(
        "--generator-dropout",
        type=float,
        help="the dropout rate in the generator",
    )
    group.add_argument(
        "--generator-lr",
        type=float,
        help="the learning rate for the generator",
    )
    group.add_argument(
        "--generator-residual",
        action="store_true",
        help="whether to use residual connections in the generator",
    )
    group.add_argument(
        "--generator-opt-betas",
        type=float,
        nargs=2,
        help="the beta values for the generator optimizer",
    )
    group.add_argument(
        "--discriminator-n-layers-hidden",
        type=int,
        help="the number of hidden layers in the discriminator",
    )
    group.add_argument(
        "--discriminator-n-units-hidden",
        type=int,
        help="the number of units in each hidden layer of the discriminator",
    )
    group.add_argument(
        "--discriminator-activation",
        type=str,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the discriminator",
    )
    group.add_argument(
        "--discriminator-batch-norm",
        action="store_true",
        help="whether to use batch norm in the discriminator",
    )
    group.add_argument(
        "--discriminator-dropout",
        type=float,
        help="the dropout rate in the discriminator",
    )
    group.add_argument(
        "--discriminator-lr",
        type=float,
        help="the learning rate for the discriminator",
    )
    group.add_argument(
        "--discriminator-opt-betas",
        type=float,
        nargs=2,
        help="the beta values for the discriminator optimizer",
    )
    group.add_argument(
        "--clipping-value",
        type=float,
        help="the clipping value for the discriminator",
    )
    group.add_argument(
        "--lambda-gradient-penalty",
        type=float,
        help="the gradient penalty coefficient",
    )


def add_tabular_gan_args(group: argparse._ArgumentGroup, overrides: bool = False) -> None:
    group.add_argument(
        "--generator-activation-out-discrete",
        type=str,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the generator output layer for discrete columns",
    )
    group.add_argument(
        "--generator-activation-out-continuous",
        type=str,
        choices=list(ACTIVATION_FUNCTIONS.keys()),
        help="the activation function of the generator output layer for continuous columns",
    )
