import argparse
import pickle
from pathlib import Path

import pandas as pd
from nhssynth.common.io import *
from nhssynth.modules.dataloader.metatransformer import MetaTransformer


class Experiments:
    def __init__(self, experiments: pd.DataFrame):
        self.contents = experiments


class SyntheticDatasets:
    def __init__(self, synthetic_datasets: pd.DataFrame):
        self.contents = synthetic_datasets


def check_input_paths(
    fn_dataset: str, fn_transformed: str, fn_metatransformer: str, dir_experiment: Path
) -> tuple[str, str]:
    """
    Sets up the input and output paths for the model files.

    Args:
        fn_dataset: The base name of the dataset.
        fn_transformed: The name of the transformed data file.
        fn_metatransformer: The name of the metatransformer file.
        dir_experiment: The path to the experiment directory.

    Returns:
        The paths to the data, metadata and metatransformer files.
    """
    fn_dataset = Path(fn_dataset).stem
    fn_transformed, fn_metatransformer = consistent_endings([fn_transformed, fn_metatransformer])
    fn_transformed, fn_metatransformer = potential_suffixes([fn_transformed, fn_metatransformer], fn_dataset)
    warn_if_path_supplied([fn_transformed, fn_metatransformer], dir_experiment)
    check_exists([fn_transformed, fn_metatransformer], dir_experiment)
    return fn_dataset, fn_transformed, fn_metatransformer


def write_data_outputs(
    experiments: pd.DataFrame,
    synthetic_datasets: pd.DataFrame,
    models: pd.DataFrame,
    fn_dataset: str,
    dir_experiment: Path,
    args: argparse.Namespace,
) -> None:
    experiments = experiments.join(
        pd.DataFrame(experiments.pop("train_config").values.tolist(), index=experiments.index)
    )
    experiments = experiments.join(
        pd.DataFrame(experiments.pop("model_config").values.tolist(), index=experiments.index)
    )

    fn_experiments, fn_synthetic_datasets = consistent_endings([args.experiments, args.synthetic_datasets])
    fn_experiments, fn_synthetic_datasets = potential_suffixes([fn_experiments, fn_synthetic_datasets], fn_dataset)
    warn_if_path_supplied([fn_experiments, fn_synthetic_datasets], dir_experiment)

    with open(dir_experiment / fn_experiments, "wb") as f:
        pickle.dump(Experiments(experiments), f)
    with open(dir_experiment / fn_synthetic_datasets, "wb") as f:
        pickle.dump(SyntheticDatasets(synthetic_datasets), f)
    (dir_experiment / "models").mkdir(parents=True, exist_ok=True)
    for i, model in models.iterrows():
        fn_model = consistent_ending(args.model, ending=".pt", suffix=f"{i[0]}_repeat_{i[1]}_config_{i[2]}")
        fn_model = potential_suffix(fn_model, fn_dataset)
        model["model"].save(dir_experiment / "models" / fn_model)


def load_required_data(
    args: argparse.Namespace, dir_experiment: Path
) -> tuple[str, pd.DataFrame, dict[str, int], MetaTransformer]:
    """
    Loads the data from `args` or from disk when the dataloader has not be run previously.

    Args:
        args: The arguments passed to the module, in this case potentially carrying the outputs of the dataloader module.
        dir_experiment: The path to the experiment directory.

    Returns:
        The data, metadata and metatransformer.
    """
    if all(x in args.module_handover for x in ["dataset", "transformed", "metatransformer"]):
        return (
            args.module_handover["dataset"],
            args.module_handover["transformed"],
            args.module_handover["metatransformer"],
        )
    else:
        fn_dataset, fn_transformed, fn_metatransformer = check_input_paths(
            args.dataset, args.transformed, args.metatransformer, dir_experiment
        )

        with open(dir_experiment / fn_transformed, "rb") as f:
            data = pickle.load(f)
        with open(dir_experiment / fn_metatransformer, "rb") as f:
            mt = pickle.load(f)

        return fn_dataset, data, mt
