from time import strftime
from typing import Final

from sdmetrics.single_table import *
from sdv.single_table import *

TIME: Final = strftime("%Y_%m_%d___%H_%M_%S")

TRACKED_METRIC_CHOICES: Final = ["ELBO", "KLD", "ReconstructionLoss", "CategoricalLoss", "NumericalLoss", "Privacy"]

SDV_SYNTHESIZER_CHOICES: Final = {
    "TVAE": TVAESynthesizer,
    "CTGAN": CTGANSynthesizer,
    "CopulaGAN": CopulaGANSynthesizer,
    "Copula": GaussianCopulaSynthesizer,
}

SDV_DETECTION_METRIC_CHOICES: Final = {
    "LogisticDetection": LogisticDetection,
    "SVCDetection": SVCDetection,
}

SDV_BINARY_METRIC_CHOICES: Final = {
    "BinaryAdaBoostClassifier": BinaryAdaBoostClassifier,
    "BinaryDecisionTreeClassifier": BinaryDecisionTreeClassifier,
    "BinaryLogisticRegression": BinaryLogisticRegression,
    "BinaryMLPClassifier": BinaryMLPClassifier,
}

SDV_MULTICLASS_METRIC_CHOICES: Final = {
    "MulticlassDecisionTreeClassifier": MulticlassDecisionTreeClassifier,
    "MulticlassMLPClassifier": MulticlassMLPClassifier,
}

SDV_REGRESSION_METRIC_CHOICES: Final = {
    "LinearRegression": LinearRegression,
    "MLPRegressor": MLPRegressor,
}

SDV_METRIC_CHOICES: Final = {
    "GMLogLikelihood": GMLogLikelihood,
    "ContingencySimilarity": ContingencySimilarity,
    "ContinuousKLDivergence": ContinuousKLDivergence,
    "CorrelationSimilarity": CorrelationSimilarity,
    "DiscreteKLDivergence": DiscreteKLDivergence,
    "BoundaryAdherence": BoundaryAdherence,
    "CategoryCoverage": CategoryCoverage,
    "CSTest": CSTest,
    "KSComplement": KSComplement,
    "MissingValueSimilarity": MissingValueSimilarity,
    "RangeCoverage": RangeCoverage,
    "StatisticSimilarity": StatisticSimilarity,
    "TVComplement": TVComplement,
    "NewRowSynthesis": NewRowSynthesis,
}

SDV_PRIVACY_METRIC_CHOICES: Final = {
    "CategoricalCAP": CategoricalCAP,
    "CategoricalGeneralizedCAP": CategoricalGeneralizedCAP,
    "CategoricalZeroCAP": CategoricalZeroCAP,
    "CategoricalKNN": CategoricalKNN,
    "CategoricalNB": CategoricalNB,
    "CategoricalRF": CategoricalRF,
    "CategoricalSVM": CategoricalSVM,
    "CategoricalEnsemble": CategoricalEnsemble,
    "NumericalLR": NumericalLR,
    "NumericalMLP": NumericalMLP,
    "NumericalSVR": NumericalSVR,
    "NumericalRadiusNearestNeighbor": NumericalRadiusNearestNeighbor,
}
