# Model Card: Variational AutoEncoder with Differential Privacy

## Model Details

The implementation of the Variational AutoEncoder (VAE) with Differential Privacy within this repository is based on work done by Dominic Danks during an NHSX Analytics Unit PhD internship [(last commit to the original SynthVAE repository: commit 88a4bdf)](https://github.com/nhsengland/SynthVAE/commit/88a4bdf613f538af45834f22d38e52312cfe24c5). This model card describes an updated and extended version of the model, by Harrison Wilde. Further information about the previous version created by Dom and its model implementation can be found in Section 5.4 of the associated [report](./reports/report.pdf).

## Model Use

### Intended Use

This model is intended for use in experimenting with differential privacy and VAEs.

## Training Data

Experiments in this repository are run against the [Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) dataset](https://biostat.app.vumc.org/wiki/Main/SupportDesc) accessed via the [pycox](https://github.com/havakv/pycox) python library. We also performed further analysis on a single table that we extracted from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/).

## Performance and Limitations

A from-scratch VAE implementation was compared against various models available within the [SDV](https://sdv.dev/) framework using a variety of quality and privacy metrics on the SUPPORT dataset. The VAE was found to be competitive with all of these models across the various metrics. Differential Privacy (DP) was introduced via [DP-SGD](https://dl.acm.org/doi/10.1145/2976749.2978318) and the performance of the VAE for different levels
of privacy was evaluated. It was found that as the level of Differential Privacy introduced by
DP-SGD was increased, it became easier to distinguish between synthetic and real data.

Proper evaluation of quality and privacy of synthetic data is challenging. In this work, we
utilised metrics from the SDV library due to their natural integration with the rest of the codebase.
A valuable extension of this work would be to apply a variety of external metrics, including
more advanced adversarial attacks to more thoroughly evaluate the privacy of the considered methods,
including as the level of DP is varied. It would also be of interest to apply DP-SGD and/or
[PATE](https://arxiv.org/pdf/1610.05755.pdf) to all of the considered methods and evaluate
whether the performance drop as a function of implemented privacy is similar or different
across the models.

### Data Handling Capabilities

The model handles continuous, categorical, and datetime variables with optimized transformations:

- **Missingness handling**: Supports both augmentation (modeling missingness as a feature) and imputation strategies (mean, median, mode)
- **Continuous variables**: Uses Bayesian Gaussian Mixture Models (GMM) with automatic component selection (1-10 components) based on variable characteristics
- **Categorical variables**: Standard one-hot encoding with support for rare categories
- **Datetime variables**: Special handling with single Gaussian component and aggressive temperature scaling to maintain wide temporal ranges
- **Constraints**: Post-generation constraint repair achieving <1% violation rates

### Data Transformation

The model uses Bayesian Gaussian Mixture Models for continuous variable transformation with the following key features:

- **Automatic component selection**: Sparse Bayesian prior (weight_concentration_prior=1e-3) enables automatic determination of optimal component count per variable
- **Kurtosis detection**: Identifies heavily-peaked distributions (excess kurtosis > 5) during transformation to inform generation strategy
- **Adaptive configuration**: Variables are automatically categorized as peaked, normal, or datetime for differential treatment during generation
- **Z-score normalization**: Optimized std_multiplier=1 for proper calibration with GMM component variances

See [config/optimized_transformer_config.yaml](../config/optimized_transformer_config.yaml) for complete transformation settings.

### Generation Process

The VAE decoder applies adaptive temperature scaling to preserve variable characteristics:

- **Peaked distributions** (high kurtosis > 5): 1.5x temperature to maintain characteristic peakedness
- **Normal distributions**: 3.0x temperature for appropriate spread
- **Datetime variables**: 15.0x temperature (3.0 × 5.0 boost) to achieve wide temporal ranges
- **GMM component softening**: 2.0x temperature on component logits to smooth boundaries
- **Post-generation smoothing**: 3% Gaussian noise applied to blur residual GMM peaks

This adaptive approach preserves both unimodal peaked variables (e.g., heavily-tailed distributions) and smooth temporal trends while preventing artificial multimodality and clipping.

### Constraint Handling

The model includes sophisticated post-generation constraint repair:

- Iterative constraint satisfaction with minimal data perturbation
- Support for inequality constraints (<, <=, >, >=), range constraints (in), and fixed combinations (fixcombo)
- Optimization to achieve <1% constraint violation rates (down from 50,000+ violations in early versions)
- Preserves statistical properties while enforcing logical relationships

See [config/IMPLEMENTATION_SUMMARY.md](../config/IMPLEMENTATION_SUMMARY.md) for detailed technical documentation of the constraint repair system and fidelity improvements.

### Current Limitations

- Hyperparameter tuning may result in errors if certain parameter values are selected (particularly extreme learning rates). Consider adjusting hyperparameters if training fails.
- Special types such as nominal data may not be handled optimally, though the model may still run
- The adaptive temperature system is optimized for the SUPPORT dataset characteristics and may require tuning for datasets with significantly different distributions

### Configuration and Examples

- **Optimized configuration**: See [config/optimized_transformer_config.yaml](../config/optimized_transformer_config.yaml) for all settings and rationale
- **Working example**: See [auxiliary/mwe_optimized.ipynb](../auxiliary/mwe_optimized.ipynb) for a complete workflow demonstrating the optimized configuration
- **Implementation details**: See [config/IMPLEMENTATION_SUMMARY.md](../config/IMPLEMENTATION_SUMMARY.md) for technical details and testing guidelines

## Acknowledgements

This documentation is inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from
Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf).
