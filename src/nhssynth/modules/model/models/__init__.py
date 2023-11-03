from typing import Final

from .dpvae import DPVAE
from .gan import GAN

# from .tabular_gan import TabularGAN
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
    "GAN": GAN,
    # "TabularGAN": TabularGAN,
}
