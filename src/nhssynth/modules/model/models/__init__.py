from typing import Final

from .dpvae import DPVAE

# GAN disabled - MLP module not implemented
# from .gan import GAN
# from .tabular_gan import TabularGAN
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
    # "GAN": GAN,  # GAN disabled - MLP module not implemented
    # "TabularGAN": TabularGAN,
}
