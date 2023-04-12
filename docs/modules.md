# Modules

This folder contains all of the modules contained in this package. They can be used together or independently - through importing them into your existing codebase or using the CLI to select which / all modules to run.

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

    In `mymodule/executor.py` and export it by adding `#!python from .executor import myexecutor` to `mymodule/__init__.py`.

3. In the `cli` folder, add a corresponding function to `arguments.py` and populate with arguments you want to expose in the CLI:

    ```python
    def add_mymodule_args(parser: argparse.ArgumentParser, group_title: str, overrides=False):
        group = parser.add_argument_group(title=group_title)
        group.add_argument(...)
        group.add_argument(...)
        ...
    ```

4. Next, in `module_setup.py` make the following adjustments the following code:

    ```python
    from nhssynth.modules import ..., mymodule, ...
    ```

    ```python hl_lines="3 4 5 6 7 8 9"
    MODULE_MAP = {
        ...
        "mymodule": ModuleConfig(
            func=mymodule.myexecutor,
            add_args_func=add_mymodule_args,
            description="...",
            help="...",
            common_parsers=[...]
        ),
        ...
    }
    ```

    Where `#!python common_parsers` is a subset of `#!python COMMON_PARSERS` defined in `common_arguments.py`. Note that the "dataset" and "core" parsers are added automatically, so you don't need to specify them. These parsers can be used to add arguments to your module that are common to multiple modules, e.g. the `dataloader` and `evaluation` modules both use `--typed` to specify the path of the typed input dataset.

5. You can (optionally) also edit the following block if you want your module to be included in a full pipeline run:

    ```python
    PIPELINE = [..., mymodule, ...]  # NOTE this determines the order of a pipeline run
    ```

6. Congrats, your module is implemented!

