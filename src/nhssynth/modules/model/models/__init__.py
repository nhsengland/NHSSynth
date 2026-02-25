from typing import Final

from .copula import Copula
from .ctgan import CTGAN
from .dpctgan import DPCTGAN
from .dpgan import DPGAN
from .dpvae import DPVAE
from .gan import GAN
from .marginal import Marginal
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
    "GAN": GAN,
    "DPGAN": DPGAN,
    "CTGAN": CTGAN,
    "DPCTGAN": DPCTGAN,
    "Marginal": Marginal,
    "Copula": Copula,
}
