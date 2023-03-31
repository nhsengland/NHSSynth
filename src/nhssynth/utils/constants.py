from time import strftime

from sdv.single_table import *

TIME = strftime("%Y_%m_%d___%H_%M_%S")

SDV_SYNTHESIZER_CHOICES = {
    "TVAE": TVAESynthesizer,
    "CTGAN": CTGANSynthesizer,
    "CopulaGAN": CopulaGANSynthesizer,
    "Copula": GaussianCopulaSynthesizer,
}
