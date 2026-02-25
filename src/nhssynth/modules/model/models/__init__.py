from typing import Final

from .ctgan import CTGAN
from .dpctgan import DPCTGAN
from .dpgan import DPGAN
from .dpvae import DPVAE
from .gan import GAN
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
    "GAN": GAN,
    "DPGAN": DPGAN,
    "CTGAN": CTGAN,
    "DPCTGAN": DPCTGAN,
}
