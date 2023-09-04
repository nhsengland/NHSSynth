from nhssynth.common.io import *


def check_input_paths(
    dir_experiment: str,
    fn_dataset: str,
    fn_typed: str,
    fn_experiments: str,
    fn_synthetic_datasets: str,
    fn_evaluations: str,
) -> str:
    """
    Sets up the input and output paths for the model files.

    Args:
        dir_experiment: The path to the experiment directory.
        fn_dataset: The base name of the dataset.
        fn_experiments: The filename of the collection of experiments.
        fn_synthetic_datasets: The filename of the collection of synthetic datasets.
        fn_evaluations: The filename of the collection of evaluations.

    Returns:
        The paths
    """
    fn_dataset = Path(fn_dataset).stem
    fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations = consistent_endings(
        [fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations]
    )
    fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations = potential_suffixes(
        [fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations], fn_dataset
    )
    warn_if_path_supplied([fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations], dir_experiment)
    check_exists([fn_typed, fn_experiments, fn_synthetic_datasets, fn_evaluations], dir_experiment)
    return (
        dir_experiment / fn_typed,
        dir_experiment / fn_experiments,
        dir_experiment / fn_synthetic_datasets,
        dir_experiment / fn_evaluations,
    )
