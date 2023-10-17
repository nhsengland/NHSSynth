import argparse
import warnings
from typing import Any

import pandas as pd
from sdmetrics.single_table import MultiColumnPairsMetric, MultiSingleColumnMetric
from tqdm import tqdm

from nhssynth.common.constants import (
    CATEGORICAL_PRIVACY_METRICS,
    METRIC_CHOICES,
    NUMERICAL_PRIVACY_METRICS,
    TABLE_METRICS,
)
from nhssynth.modules.evaluation.aequitas import run_aequitas
from nhssynth.modules.evaluation.tasks import Task, get_tasks


class EvalFrame:
    """
    Data structure for specifying and recording the evaluations of a set of synthetic datasets against a real dataset.
    All of the choices made by the user in the evaluation module are consolidated into this class.

    After running `evaluate` on a set of synthetic datasets, the evaluations can be retrieved using `get_evaluations`.
    They are stored in a dict of dataframes with indices matching that of the supplied dataframe of synthetic datasets.

    Args:
        tasks: A list of downstream tasks to run on the experiments.
        metrics: A list of metrics to calculate on the experiments.
        sdv_metadata: The SDV metadata for the dataset.
        aequitas: Whether to run Aequitas on the results of supported downstream tasks.
        aequitas_attributes: The fairness-related attributes to use for Aequitas analysis.
        key_numerical_fields: The numerical fields to use for SDV privacy metrics.
        sensitive_numerical_fields: The numerical fields to use for SDV privacy metrics.
        key_categorical_fields: The categorical fields to use for SDV privacy metrics.
        sensitive_categorical_fields: The categorical fields to use for SDV privacy metrics.
    """

    def __init__(
        self,
        tasks: list[Task],
        metrics: list[str],
        sdv_metadata: dict[str, dict[str, str]],
        aequitas: bool = False,
        aequitas_attributes: list[str] = [],
        key_numerical_fields: list[str] = [],
        sensitive_numerical_fields: list[str] = [],
        key_categorical_fields: list[str] = [],
        sensitive_categorical_fields: list[str] = [],
    ):
        self._tasks = tasks
        self._aequitas = aequitas
        self._aequitas_attributes = aequitas_attributes

        self._metrics = metrics
        self._sdv_metadata = sdv_metadata

        self._key_numerical_fields = key_numerical_fields
        self._sensitive_numerical_fields = sensitive_numerical_fields
        self._key_categorical_fields = key_categorical_fields
        self._sensitive_categorical_fields = sensitive_categorical_fields
        assert all([metric not in NUMERICAL_PRIVACY_METRICS for metric in self._metrics]) or (
            self._key_numerical_fields and self._sensitive_numerical_fields
        ), "Numerical key and sensitive fields must be provided when an SDV privacy metric is used."
        assert all([metric not in CATEGORICAL_PRIVACY_METRICS for metric in self._metrics]) or (
            self._key_categorical_fields and self._sensitive_categorical_fields
        ), "Categorical key and sensitive fields must be provided when an SDV privacy metric is used."

        self._metric_groups = self._build_metric_groups()

    def _build_metric_groups(self) -> list[str]:
        """
        Iterate through the concatenated list of metrics provided by the user and refer to the
        [defined metric groups][nhssynth.common.constants] to identify which to evaluate.

        Returns:
            A list of metric groups to evaluate.
        """
        metric_groups = set()
        if self._tasks:
            metric_groups.add("task")
        if self._aequitas:
            metric_groups.add("aequitas")
        for metric in self._metrics:
            if metric in TABLE_METRICS:
                metric_groups.add("table")
            if metric in NUMERICAL_PRIVACY_METRICS or metric in CATEGORICAL_PRIVACY_METRICS:
                metric_groups.add("privacy")
            if metric in TABLE_METRICS and issubclass(TABLE_METRICS[metric], MultiSingleColumnMetric):
                metric_groups.add("columnwise")
            if metric in TABLE_METRICS and issubclass(TABLE_METRICS[metric], MultiColumnPairsMetric):
                metric_groups.add("pairwise")
        return list(metric_groups)

    def evaluate(self, real_dataset: pd.DataFrame, synthetic_datasets: list[dict[str, Any]]) -> None:
        """
        Evaluate a set of synthetic datasets against a real dataset.

        Args:
            real_dataset: The real dataset to evaluate against.
            synthetic_datasets: The synthetic datasets to evaluate.
        """
        assert not any("Real" in i for i in synthetic_datasets.index), "Real is a reserved dataset ID."
        assert synthetic_datasets.index.is_unique, "Dataset IDs must be unique."
        self._evaluations = pd.DataFrame(index=synthetic_datasets.index, columns=self._metric_groups)
        self._evaluations.loc[("Real", None, None)] = self._step(real_dataset)
        pbar = tqdm(synthetic_datasets.iterrows(), desc="Evaluating", total=len(synthetic_datasets))
        for i, dataset in pbar:
            pbar.set_description(f"Evaluating {i[0]}, repeat {i[1]}, config {i[2]}")
            self._evaluations.loc[i] = self._step(real_dataset, dataset.values[0])

    def get_evaluations(self) -> dict[str, pd.DataFrame]:
        """
        Unpack the `self._evaluations` dataframe, where each metric group is a column, into a dict of dataframes.

        Returns:
            A dict of dataframes, one for each metric group, containing the evaluations.
        """
        assert hasattr(
            self, "_evaluations"
        ), "You must first run `evaluate` on a `real_dataset` and set of `synthetic_datasets`."
        return {
            metric_group: pd.DataFrame(
                self._evaluations[metric_group].values.tolist(), index=self._evaluations.index
            ).dropna(how="all")
            for metric_group in self._metric_groups
        }

    def _task_step(self, data: pd.DataFrame) -> dict[str, dict]:
        """
        Run the downstream tasks on the dataset. Optionally run Aequitas on the results of the tasks.

        Args:
            data: The dataset to run the tasks on.

        Returns:
            A dict of dicts, one for each metric group, to be populated with each groups metric values.
        """
        metric_dict = {metric_group: {} for metric_group in self._metric_groups}
        for task in tqdm(self._tasks, desc="Running downstream tasks", leave=False):
            task_pred_column, task_metric_values = task.run(data)
            metric_dict["task"].update(task_metric_values)
            if self._aequitas and task.supports_aequitas:
                metric_dict["aequitas"].update(run_aequitas(data[self._aequitas_attributes].join(task_pred_column)))
        return metric_dict

    def _compute_metric(
        self, metric_dict: dict, metric: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> dict[str, dict]:
        """
        Given a metric, determine the correct way to evaluate it via the lists defined in `nhssynth.common.constants`.

        Args:
            metric_dict: The dict of dicts to populate with metric values.
            metric: The metric to evaluate.
            real_data: The real dataset to evaluate against.
            synthetic_data: The synthetic dataset to evaluate.

        Returns:
            The metric_dict updated with the value of the metric.
        """
        with pd.option_context("mode.chained_assignment", None), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ConvergenceWarning")
            if metric in TABLE_METRICS:
                metric_dict["table"][metric] = TABLE_METRICS[metric].compute(
                    real_data, synthetic_data, self._sdv_metadata
                )
                if issubclass(TABLE_METRICS[metric], MultiSingleColumnMetric):
                    metric_dict["columnwise"][metric] = TABLE_METRICS[metric].compute_breakdown(
                        real_data, synthetic_data, self._sdv_metadata
                    )
                elif issubclass(TABLE_METRICS[metric], MultiColumnPairsMetric):
                    metric_dict["pairwise"][metric] = TABLE_METRICS[metric].compute_breakdown(
                        real_data, synthetic_data, self._sdv_metadata
                    )
            elif metric in NUMERICAL_PRIVACY_METRICS:
                metric_dict["privacy"][metric] = NUMERICAL_PRIVACY_METRICS[metric].compute(
                    real_data.dropna(),
                    synthetic_data.dropna(),
                    self._sdv_metadata,
                    self._key_numerical_fields,
                    self._sensitive_numerical_fields,
                )
            elif metric in CATEGORICAL_PRIVACY_METRICS:
                metric_dict["privacy"][metric] = CATEGORICAL_PRIVACY_METRICS[metric].compute(
                    real_data.dropna(),
                    synthetic_data.dropna(),
                    self._sdv_metadata,
                    self._key_categorical_fields,
                    self._sensitive_categorical_fields,
                )
        return metric_dict

    def _step(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame = None) -> dict[str, dict]:
        """
        Run the two functions above (or only the tasks when no synthetic data is provided).

        Args:
            real_data: The real dataset to evaluate against.
            synthetic_data: The synthetic dataset to evaluate.

        Returns:
            A dict of dicts, one for each metric grou, to populate a row of `self._evaluations` corresponding to the `synthetic_data`.
        """
        if synthetic_data is None:
            metric_dict = self._task_step(real_data)
        else:
            metric_dict = self._task_step(synthetic_data)
            for metric in tqdm(self._metrics, desc="Running metrics", leave=False):
                metric_dict = self._compute_metric(metric_dict, metric, real_data, synthetic_data)
        return metric_dict


def validate_metric_args(
    args: argparse.Namespace, fn_dataset: str, columns: pd.Index
) -> tuple[list[Task], argparse.Namespace]:
    """
    Validate the arguments for downstream tasks and Aequitas.

    Args:
        args: The argument namespace to validate.
        fn_dataset: The name of the dataset.
        columns: The columns in the dataset.

    Returns:
        The validated arguments, the list of tasks and the list of metrics.
    """
    if args.downstream_tasks:
        tasks = get_tasks(fn_dataset, args.tasks_dir)
        if not tasks:
            warnings.warn("No valid downstream tasks found.")
    else:
        tasks = []
    if args.aequitas:
        if not args.downstream_tasks or not any([task.supports_aequitas for task in tasks]):
            warnings.warn(
                "Aequitas can only work in context of downstream tasks involving binary classification problems."
            )
        if not args.aequitas_attributes:
            warnings.warn("No attributes specified for Aequitas analysis, defaulting to all columns in the dataset.")
            args.aequitas_attributes = columns.tolist()
        assert all(
            [attr in columns for attr in args.aequitas_attributes]
        ), "Invalid attribute(s) specified for Aequitas analysis."
    metrics = {}
    for metric_group in METRIC_CHOICES:
        selected_metrics = getattr(args, "_".join(metric_group.split()).lower() + "_metrics") or []
        metrics.update({metric_name: METRIC_CHOICES[metric_group][metric_name] for metric_name in selected_metrics})
    return args, tasks, metrics
