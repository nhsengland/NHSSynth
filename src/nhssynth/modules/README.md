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
2. Include within it a main executor that accepts arguments from the CLI, e.g.

    ```python
    def myexecutor(args):
        ...
    ```

    In `mymodule/executor.py` and export this by adding `from .executor import myexecutor` in `mymodule/__init__.py`.

3. In the `cli` folder, add the following code blocks to `arguments.py` and populate them in a similar fashion to the other modules as you build:

    ```python
    def add_mymodule_args(parser: argparse.ArgumentParser):
        ...
    ```

    The following code blocks are optional, ddd them if this module should be executed as part of a full pipeline run:

    ```python
    def add_all_module_args(parser: argparse.ArgumentParser):
        ...
        mymodule_group = parser.add_argument_group(title="mymodule")
        add_mymodule_args(mymodule_group)
        ...
    ```
    
    ```python
    def add_config_args(parser: argparse.ArgumentParser, override=False):
        ...
        add_mymodule_args(overrides_group)
        ...
    ```

4. Mext, in `module_setup.py` add the following code:

    ```python
    from nhssynth.modules import ..., mymodule, ...
    ```

    ```python
    MODULE_MAP = {
        ...
        "mymodule": ModuleConfig(
            mymodule.myexecutor,
            add_mymodule_args,
            "<description>",
            "<short help>",
        ),
        ...
    }
    ```

    And again, edit the following block if you want your module to be included in a full pipeline run:

    ```python
    def run_pipeline(args):
        ...
        mymodule.myexecutor(args)
        ...
    ```

5. Finally, add the following line of code to `run.py`:

    ```python
    def run()
        ...
        add_module_subparser(subparsers, "mymodule")
        ...
    ```

6. Congrats, your module is implemented!

