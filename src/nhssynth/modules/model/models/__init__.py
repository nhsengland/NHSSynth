from typing import Final

from .dpgan import DPGAN
from .dpvae import DPVAE
from .gan import GAN
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
    "GAN": GAN,
    "DPGAN": DPGAN,
}
