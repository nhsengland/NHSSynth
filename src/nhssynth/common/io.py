"""Common building-block functions for handling module input and output."""
import warnings
from pathlib import Path
from typing import Union


def experiment_io(experiment_name: str, dir_experiments: str = "experiments") -> str:
    """
    Create an experiment's directory and return the path.

    Args:
        experiment_name: The name of the experiment.
        dir_experiments: The name of the directory containing all experiments.

    Returns:
        The path to the experiment directory.
    """
    dir_experiment = Path(dir_experiments) / experiment_name
    dir_experiment.mkdir(parents=True, exist_ok=True)
    return dir_experiment


def consistent_ending(fn: str, ending: str = ".pkl", suffix: str = "") -> str:
    """
    Ensures that the filename `fn` ends with `ending`. If not, removes any existing ending and appends `ending`.

    Args:
        fn: The filename to check.
        ending: The desired ending to check for. Default is ".pkl".
        suffix: A suffix to append to the filename before the ending.

    Returns:
        The filename with the correct ending and potentially an inserted suffix.
    """
    path_fn = Path(fn)
    return str(path_fn.parent / path_fn.stem) + ("_" if suffix else "") + suffix + ending


def consistent_endings(args: list[Union[str, tuple[str, str], tuple[str, str, str]]]) -> list[str]:
    return list(consistent_ending(arg) if isinstance(arg, str) else consistent_ending(*arg) for arg in args)


def potential_suffix(fn: str, fn_base: str) -> str:
    """
    Checks if `fn` is a suffix (starts with an underscore) to append to `fn_base`, or a filename in its own right.

    Args:
        fn: The filename / potential suffix to append to `fn_base`.
        fn_base: The name of the file the suffix would attach to.

    Returns:
        The appropriately processed `fn`
    """
    fn_base = Path(fn_base).stem
    if fn[0] == "_":
        return fn_base + fn
    else:
        return fn


def potential_suffixes(fns: list[str], fn_base: str) -> list[str]:
    return list(potential_suffix(fn, fn_base) for fn in fns)


def check_exists(fns: list[str], dir: Path) -> None:
    """
    Checks if the files in `fns` exist in `dir`.

    Args:
        fns: The list of files to check.
        dir: The directory the files should exist in.

    Raises:
        FileNotFoundError: If any of the files in `fns` do not exist in `dir`.
    """
    for fn in fns:
        if not (dir / fn).exists():
            raise FileNotFoundError(f"File {fn} does not exist at {dir}.")


def warn_if_path_supplied(fns: list[str], dir: Path) -> None:
    """
    Warns if the files in `fns` include directory separators.

    Args:
        fns: The list of files to check.
        dir: The directory the files should exist in.

    Warnings:
        UserWarning: when the path to any of the files in `fns` includes directory separators, as this may lead to unintended consequences if the user doesn't realise default directories are pre-specified.
    """
    for fn in fns:
        if "/" in fn:
            warnings.warn(
                f"Using the path supplied appended to {dir}, i.e. attempting to read data from {dir / fn}",
                UserWarning,
            )
