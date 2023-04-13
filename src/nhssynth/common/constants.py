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

SDV_COLUMN_SHAPE_METRIC_CHOICES: Final = {
    "KSComplement": KSComplement,
    "TVComplement": TVComplement,
    "CSTest": CSTest,
}

SDV_COLUMN_SIMILARITY_METRIC_CHOICES: Final = {
    "MissingValueSimilarity": MissingValueSimilarity,
    "StatisticSimilarity": StatisticSimilarity,
    "CorrelationSimilarity": CorrelationSimilarity,
    "ContingencySimilarity": ContingencySimilarity,
}

SDV_COVERAGE_METRIC_CHOICES: Final = {
    "RangeCoverage": RangeCoverage,
    "CategoryCoverage": CategoryCoverage,
}

SDV_SYNTHESIS_METRIC_CHOICES: Final = {
    "NewRowSynthesis": NewRowSynthesis,
}

SDV_BOUNDARY_METRIC_CHOICES: Final = {
    "BoundaryAdherence": BoundaryAdherence,
}

SDV_DIVERGENCE_METRIC_CHOICES: Final = {
    "ContinuousKLDivergence": ContinuousKLDivergence,
    "DiscreteKLDivergence": DiscreteKLDivergence,
}

SDV_CAP_PRIVACY_METRIC_CHOICES: Final = {
    "CategoricalCAP": CategoricalCAP,
    "CategoricalGeneralizedCAP": CategoricalGeneralizedCAP,
    "CategoricalZeroCAP": CategoricalZeroCAP,
}

SDV_CATEGORICAL_PRIVACY_METRIC_CHOICES: Final = {
    "CategoricalKNN": CategoricalKNN,
    "CategoricalNB": CategoricalNB,
    "CategoricalRF": CategoricalRF,
    "CategoricalSVM": CategoricalSVM,
    "CategoricalEnsemble": CategoricalEnsemble,
}

SDV_NUMERICAL_PRIVACY_METRIC_CHOICES: Final = {
    "NumericalLR": NumericalLR,
    "NumericalMLP": NumericalMLP,
    "NumericalSVR": NumericalSVR,
    "NumericalRadiusNearestNeighbor": NumericalRadiusNearestNeighbor,
}

SDV_METRIC_CHOICES: Final = {
    "Detection": SDV_DETECTION_METRIC_CHOICES,
    "Binary": SDV_BINARY_METRIC_CHOICES,
    "Multiclass": SDV_MULTICLASS_METRIC_CHOICES,
    "Regression": SDV_REGRESSION_METRIC_CHOICES,
    "Column Shape": SDV_COLUMN_SHAPE_METRIC_CHOICES,
    "Column Similarity": SDV_COLUMN_SIMILARITY_METRIC_CHOICES,
    "Coverage": SDV_COVERAGE_METRIC_CHOICES,
    "Synthesis": SDV_SYNTHESIS_METRIC_CHOICES,
    "Boundary": SDV_BOUNDARY_METRIC_CHOICES,
    "Divergence": SDV_DIVERGENCE_METRIC_CHOICES,
    "CAP Privacy": SDV_CAP_PRIVACY_METRIC_CHOICES,
    "Categorical Privacy": SDV_CATEGORICAL_PRIVACY_METRIC_CHOICES,
    "Numerical Privacy": SDV_NUMERICAL_PRIVACY_METRIC_CHOICES,
}
