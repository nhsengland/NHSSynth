from time import strftime
from typing import Final

from sdv.single_table import *

TIME: Final = strftime("%Y_%m_%d___%H_%M_%S")

SDV_SYNTHESIZER_CHOICES: Final = {
    "TVAE": TVAESynthesizer,
    "CTGAN": CTGANSynthesizer,
    "CopulaGAN": CopulaGANSynthesizer,
    "Copula": GaussianCopulaSynthesizer,
}
