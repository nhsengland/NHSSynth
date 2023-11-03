# Adding new models

The `model` module contains all of the architectures implemented as part of this package. We offer GAN and VAE based architectures with a number of adjustments to achieve privacy and other augmented functionalities. The module handles the training and generation of synthetic data using these architectures, per a user's choice of model(s) and configuration.

It is likely that as the literature matures, more effective architectures will present themselves as promising for application to the type of tabular data `NHSSynth` is designed for. Below we discuss how to add new models to the package.

## Model design

In general, we view the VAE and (Tabular)GAN implementations in this package as the foundations for other architectures. As such, we try to maintain a somewhat modular design to building up more complex differentially private and so on architectures. Each model inherits from either the `GAN` or `VAE` class ([in files of the same name](https://github.com/nhsengland/NHSSynth/tree/main/src/nhssynth/modules/model/models))
