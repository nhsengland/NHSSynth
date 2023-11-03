# Adding new models

The `model` module contains all of the architectures implemented as part of this package. We offer GAN and VAE based architectures with a number of adjustments to achieve privacy and other augmented functionalities. The module handles the training and generation of synthetic data using these architectures, per a user's choice of model(s) and configuration.

It is likely that as the literature matures, more effective architectures will present themselves as promising for application to the type of tabular data `NHSSynth` is designed for. Below we discuss how to add new models to the package.

## Model design

The models in this package are built entirely in [PyTorch](https://pytorch.org) and use [Opacus](https://opacus.ai) for differential privacy.

We have built the VAE and (Tabular)GAN implementations in this package to serve as the foundations for a number of other architectures. As such, we try to maintain a somewhat modular design to building up more complex differentially private (or otherwise augmented) architectures. Each model inherits from either the `GAN` or `VAE` class ([in files of the same name](https://github.com/nhsengland/NHSSynth/tree/main/src/nhssynth/modules/model/models)) which in turn inherit from a generic `Model` class found in the [`common`](https://github.com/nhsengland/NHSSynth/tree/main/src/nhssynth/modules/model/common) folder. This folder contains components of models which are not to be instantiated themselves, e.g. a mixin class for differential privacy, the MLP underlying the `GAN` and so on.

The `Model` class from which all of the models derive handles all of the general attributes. Roughly, these are the specifics of the dataset the instance of the model is relative to, the device that training is to be carried out upon, and other training parameters such as the total number of epochs to execute.

We define these things at the model level, as when using differential privacy or other privacy accountant methods, we must know ahead of time the data and length of training exposure in order to calculate the levels of noise required to reach a certain privacy guarantee and so on.

## Implementing a new model

In order to add a new architecture then, it is important to first investigate the modular parts already implemented to ensure that what you want to build is not already possible through the composition of these existing parts. Then you must ensure that your architecture either inherits from the `GAN` or `VAE`, or `Model` if you wish to implement a different type of generative model.

In all of these cases, the interface expects for the implementation to have the following methods:

- `get_args`: a class method that lists the architecture specific arguments that the model requires. This is used to facilitate default arguments in the python API whilst still allowing for arguments in the CLI to be propagated and recorded automatically in the experiment output. This should be a list of variable names equal to the concatenation of all of the non-`Model` parent classes (e.g. `DPVAE` has `DP` and `VAE` args) plus any architecture specific arguments in the `__init__` method of the model in question.
- `get_metrics`: another class method that behaves similarly to the above, should return a list of valid metrics to track during training for this model
- `train`: a method handling the training loop for the model. This should take `num_epochs`, `patience` and `displayed_metrics` as arguments and return a tuple containing the number of epochs that were executed plus a bundle of training metrics (the values over time returned by `get_metrics` on the class). In the execution of this method, the utility methods defined in `Model` should be called in order, `_start_training` at the beginning, then `_record_metrics` at each training step of the data loader, and finally `_finish_training` to clean up progress bars and so on. `displayed_metrics` determines which metrics are actively displayed during training.
- `generate`: a method to call on the trained model which generates `N` samples of data, and calls the model's associated `MetaTransformer` to return a valid pandas DataFrame of synthetic data ready to output.

## Adding a new model to the CLI

Once you have implemented your new model, you must add it to the CLI. To do this, we must first export the model's class into the `MODELS` constant in the `__init__` file [in the `models` subfolder](https://github.com/nhsengland/NHSSynth/blob/main/src/nhssynth/modules/model/models/__init__.py). We can then add a new function and option in [`module_arguments.py`](https://github.com/nhsengland/NHSSynth/blob/main/src/nhssynth/cli/model_arguments.py) to list the arguments and their types unique to this type of architecture.

*Note that you should not duplicate arguments that are already defined in the `Model` class or foundational model architectures such as the `GAN` if you are implementing an extension to it. If you have setup `get_args` correctly all of this will be propagated automatically.*
