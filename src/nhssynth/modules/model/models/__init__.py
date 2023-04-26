from typing import Final

from .DPVAE import DPVAE
from .VAE import VAE

MODELS: Final = {
    "VAE": VAE,
    "DPVAE": DPVAE,
}
