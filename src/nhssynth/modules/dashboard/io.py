from nhssynth.common.io import *


def check_input_paths(
    dir_experiment: str, fn_dataset: str, fn_typed: str, fn_experiments: str, fn_evaluation_bundle: str
) -> str:
    """
    Sets up the input and output paths for the model files.

    Args:
        dir_experiment: The path to the experiment directory.
        fn_dataset: The base name of the dataset.
        fn_experiments: The name of the set of experiments.
        fn_experiment_bundle: The name of the experiment bundle file.

    Returns:
        The path to output the model.
    """
    fn_dataset = Path(fn_dataset).stem
    fn_typed, fn_experiments, fn_evaluation_bundle = consistent_endings(
        [fn_typed, fn_experiments, fn_evaluation_bundle]
    )
    fn_typed, fn_experiments, fn_evaluation_bundle = potential_suffixes(
        [fn_typed, fn_experiments, fn_evaluation_bundle], fn_dataset
    )
    warn_if_path_supplied([fn_typed, fn_experiments, fn_evaluation_bundle], dir_experiment)
    check_exists([fn_typed, fn_experiments, fn_evaluation_bundle], dir_experiment)
    return dir_experiment / fn_typed, dir_experiment / fn_experiments, dir_experiment / fn_evaluation_bundle
