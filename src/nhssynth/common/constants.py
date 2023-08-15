"""Define all of the common constants used throughout the project."""
from time import strftime
from typing import Final

import torch.nn as nn
from sdmetrics.single_table import *

TIME: Final = strftime("%Y_%m_%d___%H_%M_%S")

TRACKED_METRICS: Final = [
    "ELBO",
    "KLD",
    "ReconstructionLoss",
    "CategoricalLoss",
    "NumericalLoss",
    "Privacy",
]

ACTIVATION_FUNCTIONS: Final = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

SDV_COLUMN_SHAPE_METRICS: Final = {
    "KSComplement": KSComplement,
    "TVComplement": TVComplement,
}

SDV_COLUMN_SIMILARITY_METRICS: Final = {
    "MissingValueSimilarity": MissingValueSimilarity,
    "StatisticSimilarity": StatisticSimilarity,
    "CorrelationSimilarity": CorrelationSimilarity,
    "ContingencySimilarity": ContingencySimilarity,
}

SDV_COVERAGE_METRICS: Final = {
    "RangeCoverage": RangeCoverage,
    "CategoryCoverage": CategoryCoverage,
}

SDV_SYNTHESIS_METRICS: Final = {
    "NewRowSynthesis": NewRowSynthesis,
}

SDV_BOUNDARY_METRICS: Final = {
    "BoundaryAdherence": BoundaryAdherence,
}

SDV_DIVERGENCE_METRICS: Final = {
    "ContinuousKLDivergence": ContinuousKLDivergence,
    "DiscreteKLDivergence": DiscreteKLDivergence,
}

SDV_CAP_PRIVACY_METRICS: Final = {
    "CategoricalCAP": CategoricalCAP,
    "CategoricalGeneralizedCAP": CategoricalGeneralizedCAP,
    "CategoricalZeroCAP": CategoricalZeroCAP,
}

SDV_CATEGORICAL_PRIVACY_METRICS: Final = {
    "CategoricalKNN": CategoricalKNN,
    "CategoricalNB": CategoricalNB,
    "CategoricalRF": CategoricalRF,
    "CategoricalSVM": CategoricalSVM,
    "CategoricalEnsemble": CategoricalEnsemble,
}

SDV_NUMERICAL_PRIVACY_METRICS: Final = {
    "NumericalLR": NumericalLR,
    "NumericalMLP": NumericalMLP,
    "NumericalSVR": NumericalSVR,
    "NumericalRadiusNearestNeighbor": NumericalRadiusNearestNeighbor,
}

TABLE_METRICS: Final = {
    **SDV_COLUMN_SHAPE_METRICS,
    **SDV_COLUMN_SIMILARITY_METRICS,
    **SDV_COVERAGE_METRICS,
    **SDV_SYNTHESIS_METRICS,
    **SDV_BOUNDARY_METRICS,
    **SDV_DIVERGENCE_METRICS,
}

CATEGORICAL_PRIVACY_METRICS: Final = {
    **SDV_CAP_PRIVACY_METRICS,
    **SDV_CATEGORICAL_PRIVACY_METRICS,
}

NUMERICAL_PRIVACY_METRICS: Final = {
    **SDV_NUMERICAL_PRIVACY_METRICS,
}

METRIC_CHOICES: Final = {
    "Column Shape": SDV_COLUMN_SHAPE_METRICS,
    "Column Similarity": SDV_COLUMN_SIMILARITY_METRICS,
    "Coverage": SDV_COVERAGE_METRICS,
    "Synthesis": SDV_SYNTHESIS_METRICS,
    "Boundary": SDV_BOUNDARY_METRICS,
    "Divergence": SDV_DIVERGENCE_METRICS,
    "CAP Privacy": SDV_CAP_PRIVACY_METRICS,
    "Categorical Privacy": SDV_CATEGORICAL_PRIVACY_METRICS,
    "Numerical Privacy": SDV_NUMERICAL_PRIVACY_METRICS,
}
