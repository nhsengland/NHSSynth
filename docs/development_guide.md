# Development Guide

*This document aims to provide a comprehensive set of instructions for continuing development of this package. Good knowledge of Python development is assumed. Some ways of working are subjective and preferential; as such we try to be as minimal in our proscription of other methods as possible.*

## Dev Environment Setup

### Python

The package currently supports major versions 3.9, 3.10 and 3.11 of Python. We recommend installing all of these versions, though at minimum the latest supported version should be used. Many people use [`pyenv`](https://github.com/pyenv/pyenv) for managing multiple python versions, on MacOS [homebrew](https://brew.sh) is a good, less invasive option for this (provided you then use a virtual environment manager too). For virtual environment management, we recommend [`venv`](https://docs.python.org/3/library/venv.html), but [conda](https://docs.conda.io/en/latest/) or some other similar system would suffice (note that in the section below it may not be necessary to use any specific virtual environment management at all).

### Poetry

We use [Poetry](https://python-poetry.org/) to manage dependencies and the actual packaging and publishing of `NHSSynth` to [PyPI](https://pypi.org). Poetry is a more robust alternative to a `requirements.txt` file, allowing for grouped dependencies and advanced build options. Rather than freezing a specific `pip` state, Poetry only specifies the top-level dependencies and allows for the installation of the latest compatible versions of the full dependency tree. *See the [`pyproject.toml`](../pyproject.toml) in the GitHub repository and Poetry's documentation for further context.*

Once Poetry is installed (in your preferred way per the instructions [on their website](https://python-poetry.org/docs/#installation)), you can choose one of two options:

1. Allow `poetry` to control virtual environments in their [proprietary way](https://python-poetry.org/docs/managing-environments/)), such that when you install and develop the package poetry will automatically create a virtual environment for you.
2. Change `poetry`'s configuration to manage your own virtual environments:
      
    ```bash
    poetry config virtualenvs.create false
    poetry config virtualenvs.in-project false
    ```

    In this setup, a virtual environment can be be instantiated and activated in whichever way you prefer. For example, using `venv`:

    ```bash
    python3.11 -m venv nhssynth-3.11
    source nhssynth-3.11/bin/activate
    ```


### Package Installation

At this point, the project dependencies can be installed via `poetry install --with dev` (add optional flags: `--with aux` to work with the [auxiliary notebooks](../auxiliary/), `--with docs` to work with the [documentation](../docs/)). This will install the package in editable mode, meaning that changes to the source code will be reflected in the installed package without needing to reinstall it. *Note that if you are using your own virtual environment, you will need to activate it before running this command.*

You can then interact with the package in one of two ways:

1. Via the [CLI](src/nhssynth/cli/) module, which is accessed using the `nhssynth` command, e.g.
    
    ```bash
    poetry run nhssynth ...
    ```
    
    *Note that you can omit the `poetry run` part and just type `nhssynth` if you followed the optional steps above to manage and activate your own virtual environment, or if you have executed `poetry shell` beforehand.*
2. Through directly importing parts of the package to use in an existing project (`from nhssynth.modules... import ...`).

#### Secure Mode

Note that in order to train a generator in *secure mode* (see the [documentation](https://nhsx.github.io/NHSSynth/secure_mode/) for details) the PyTorch extension package [`csprng`](https://github.com/pytorch/csprng) must be installed separately. Currently this package's dependencies are not compatible with recent versions of PyTorch (the author's plan on rectifying this - watch this space), so you will need to install it manually:

```bash
git clone git@github.com:pytorch/csprng.git
cd csprng
git branch release "v0.2.2-rc1"
git checkout release
python setup.py install
```

## Coding Practices

### Style

We use [`black`](https://black.readthedocs.io/en/stable/) for code formatting. This is a fairly opinionated formatter, but it is widely used and has a good reputation. We also use [`isort`](https://pycqa.github.io/isort/) to manage imports. Both of these tools are run automatically via a `pre-commit` hook. Ensure you have installed the package with the `dev` group of dependencies and then run the following command to install the hooks:

```bash
pre-commit install
```

*Note that you may need to pre-pend this command with `poetry run` if you are not using your own virtual environment.*

`black` and `isort` are also run as part of the CI workflow discussed below.

### Documentation

There should be Google-style docstrings on all non-trivial functions and classes. Ideally a docstring should take the form:

```python
def func(arg1: type1, arg2: type2) -> returntype:
    """
    One-line summary of the function.
    AND / OR
    Longer description of the function, including any caveats or assumptions.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of the return value.
    """
    ...
```

These docstrings are then [compiled](../docs/scripts/gen_ref_pages.py) into a full API documentation tree as part of a larger MkDocs documentation site hosted via GitHub. This process is derived from [this tutorial](https://squidfunk.github.io/mkdocs-material/getting-started/).

The MkDocs page is built using the `mkdocs-material` theme. The documentation is built and hosted [automatically](../.github/workflows/build_mkdocs.yaml) via GitHub Pages.

The other parts of this site comprise markdown documents in the [docs](../docs/) folder. Adding new pages is handled in the [`mkdocs.yml`](../mkdocs.yml) file as in any other Material MkDocs site. See [their documentation](https://squidfunk.github.io/mkdocs-material/setup/) if more complex changes to the site are required.

## Testing

We use [`tox`](https://tox.readthedocs.io/en/latest/) to manage the testing suite. This allows us to test the package against multiple versions of Python, and to run the tests in a clean environment. To run the tests, simply execute `tox` in the root directory of the repository. This will run the tests against all supported versions of Python. To run the tests against a specific version of Python, use `tox -e py39` (or `py310` or `py311`).

### Configuration

See the [tox.ini](../tox.ini) file for more information on the testing configuration. We follow the [Poetry documentation on `tox` support](https://python-poetry.org/docs/faq/#usecase-2) to ensure that for each version of Python, `tox` will create an `sdist` package of the project and use `pip` to install it in a fresh environment. Thus, dependencies are resolved by pip in the first place and then afterwards updated to the locked dependencies in [`poetry.lock`](../poetry.lock) by running `poetry install ...`. The tests are then run using `poetry pytest`, which is configured in the [pyproject.toml](../pyproject.toml) file. This configuration is fairly minimal: simply specifying the testing directory as the [tests](../tests/) folder and filtering some known warnings.

### Adding Tests

We use the `pytest` framework for testing. The testing directory structure mirrors that of [`src`](../src/). The usual testing practices apply.

## Releases

### Version Management

The package's version should be updated following the [semantic versioning](https://semver.org/spec/v2.0.0.html) framework. The package is currently in a *pre-release* state, such that major version 1.0.0 should only be reached once the package is functionally complete and stable.

To update the package, we can use Poetry's `version` command:

```bash
poetry version <version>
```

We can then commit and push the changes to the version file, and create a new tag:

```bash
git add pyproject.toml
git commit -m "Bump version to <version>"
git push
```

We should then tag the release using GitHub's CLI:

```bash
gh release create <version> --generate-notes
```

This will create a new release on GitHub, and will automatically generate a changelog based on the commit messages and PR's closed since the last release. This changelog can then be edited to add more detail if necessary.

### Building and Publishing to PyPI

As mentioned above, Poetry offers not only dependency management, but also a simple way to build and distribute the package.

To build the package, we can use the `build` command:

```bash
poetry build
```

This will create a `dist` folder containing the built package. To publish this to PyPI, we can use the `publish` command:

```bash
poetry publish
```

This will prompt for PyPI credentials, and then publish the package. *Note that this will only work if you have been added as a Maintainer of the package on PyPI.*

*It might be preferable at some point in the future to set up Trusted Publisher Management via OpenID Connect (OIDC) to allow for automated publishing of the package via a GitHub workflow. See the "Publishing" tab of `NHSSynth`'s project management panel on PyPI to set this up.*

## Continuous Integration

We use GitHub Actions for continuous integration. The configuration for this can be found in the [`.github/workflows`](../.github/workflows/) folder. The CI workflow is triggered on every push to the `main` branch, and runs the tests against all supported versions of Python. It also runs `black` and `isort` to check that the code is formatted correctly, and builds the documentation site.

There are also scripts to update the dynamic badges in the [`README`](../README.md). These work via a gist associated with the repository. It is not easy to transfer ownership of this process, so if they break please feel free to [contact me](mailto:h.wilde@ucl.ac.uk).

## Security and Vulnerability Management

The GitHub repository for the package has Dependabot, code scanning, and other security features enabled. These should be monitored continuously and any issues resolved as soon as possible. When issues of this type require a specific version of a dependency to be specified, the version should be updated as part of the `security` group of dependencies (i.e. with `poetry add <package> --group security`) and a new release created (see above).
