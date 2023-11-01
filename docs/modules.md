# Adding new modules

The package is designed such that each module can be used as part of a pipeline (via the CLI or a configuration file) or independently (via importing them into an existing codebase).

In the future it may be desireable to add / adjust the modules of the package, this guide offers a high-level overview of how to do so.

## Importing a module from this package

After installing the package, you can simply do:
```python
from nhssynth.modules import <module>
```
and you will be able to use it in your code!

## Creating a new module and folding it into the CLI

The following instructions specify how to extend this package with a new module:

1. Create a folder for your module within the package, i.e. `src/nhssynth/modules/mymodule`
2. Include within it a main executor function that accepts arguments from the CLI, i.e.

    ```python
    def myexecutor(args):
        ...
    ```

    In `mymodule/executor.py` and export it by adding `#!python from .executor import myexecutor` to `mymodule/__init__.py`. Check the existing modules for examples of what a typical executor function looks like.

3. In the `cli` folder, add a corresponding function to `module_arguments.py` and populate with arguments you want to expose in the CLI:

    ```python
    def add_mymodule_args(parser: argparse.ArgumentParser, group_title: str, overrides=False):
        group = parser.add_argument_group(title=group_title)
        group.add_argument(...)
        group.add_argument(...)
        ...
    ```

4. Next, in `module_setup.py` make the following adjustments to the `#!python MODULE_MAP` code:

    ```python hl_lines="3 4 5 6 7 8 9"
    MODULE_MAP = {
        ...
        "mymodule": ModuleConfig(
            func=m.mymodule.myexecutor,
            add_args=ma.add_mymodule_args,
            description="...",
            help="...",
            common_parsers=[...]
        ),
        ...
    }
    ```

    Where `#!python common_parsers` is a subset of `#!python COMMON_PARSERS` defined in `common_arguments.py`. Note that the "seed" and "core" parsers are added automatically, so you don't need to specify them. These parsers can be used to add arguments to your module that are common to multiple modules, e.g. the `dataloader` and `evaluation` modules both use `--typed` to specify the path of the typed input dataset.

5. You can (optionally) also edit the following block if you want your module to be included in a full pipeline run:

    ```python
    PIPELINE = [..., mymodule, ...]  # NOTE this determines the order of a pipeline run
    ```

6. Congrats, your module is implemented within the CLI, its documentation etc. will now be built automatically and it can be referenced in configuration files!
