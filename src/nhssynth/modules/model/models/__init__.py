from typing import Final

from .dpvae import DPVAE
from .vae import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
}
