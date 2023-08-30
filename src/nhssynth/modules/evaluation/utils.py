import argparse
import warnings
from typing import Any

import pandas as pd
from nhssynth.common.constants import (
    CATEGORICAL_PRIVACY_METRICS,
    METRIC_CHOICES,
    NUMERICAL_PRIVACY_METRICS,
    TABLE_METRICS,
)
from nhssynth.common.dicts import filter_dict
from nhssynth.modules.evaluation.aequitas import run_aequitas
from nhssynth.modules.evaluation.tasks import Task, get_tasks
from sdmetrics.single_table import MultiColumnPairsMetric, MultiSingleColumnMetric
from tqdm import tqdm


class EvalFrame:
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
        self.tasks = tasks
        self.aequitas = aequitas
        self.aequitas_attributes = aequitas_attributes

        self.metrics = metrics
        self.sdv_metadata = sdv_metadata

        self.key_numerical_fields = key_numerical_fields
        self.sensitive_numerical_fields = sensitive_numerical_fields
        self.key_categorical_fields = key_categorical_fields
        self.sensitive_categorical_fields = sensitive_categorical_fields
        assert all([metric not in NUMERICAL_PRIVACY_METRICS for metric in self.metrics]) or (
            self.key_numerical_fields and self.sensitive_numerical_fields
        ), "Numerical key and sensitive fields must be provided when an SDV privacy metric is used."
        assert all([metric not in CATEGORICAL_PRIVACY_METRICS for metric in self.metrics]) or (
            self.key_categorical_fields and self.sensitive_categorical_fields
        ), "Categorical key and sensitive fields must be provided when an SDV privacy metric is used."

        self.metric_groups = self._build_metric_groups()

    def _build_metric_groups(self) -> set[str]:
        metric_groups = set()
        if self.tasks:
            metric_groups.add("task")
        if self.aequitas:
            metric_groups.add("aequitas")
        for metric in self.metrics:
            if metric in TABLE_METRICS:
                metric_groups.add("table")
            if metric in NUMERICAL_PRIVACY_METRICS or metric in CATEGORICAL_PRIVACY_METRICS:
                metric_groups.add("privacy")
            if metric in TABLE_METRICS and issubclass(TABLE_METRICS[metric], MultiSingleColumnMetric):
                metric_groups.add("columnwise")
            if metric in TABLE_METRICS and issubclass(TABLE_METRICS[metric], MultiColumnPairsMetric):
                metric_groups.add("pairwise")
        return metric_groups

    def evaluate(self, real_dataset: pd.DataFrame, experiments: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
        assert "Real" not in [experiment["id"] for experiment in experiments], "Real is a reserved experiment ID."
        self._experiments = pd.DataFrame(
            [{"architecture": "Real", "id": "Real"}]
            + [
                {
                    **filter_dict(row, {"model_config", "dataset", "num_configs"}),
                    **row["model_config"],
                }
                for row in experiments
            ]
        )
        assert len(self._experiments["id"]) == len(self._experiments["id"].unique()), "Experiment IDs must be unique."
        self._dict = {id: {} for id in self._experiments["id"]}
        self._step(real_dataset, "Real")
        pbar = tqdm(experiments, desc="Evaluating")
        for experiment in pbar:
            pbar.set_description(
                f"Evaluating {experiment['architecture']}, config {experiment['config_idx']}, repeat {experiment['repeat']}"
            )
            self._step(real_dataset, experiment["id"], experiment["dataset"])

    def get_evaluations(self) -> dict[str, pd.DataFrame]:
        """
        Return a dict of dataframes each with one column for seed, one for architecture, and one per metric / report
        """
        assert hasattr(self, "_dict"), "You must first run `evaluate` on a `real_dataset` and set of `experiments`."
        out = {}
        for metric_group in self.metric_groups:
            out[metric_group] = []
            for id in self._experiments["id"]:
                if id != "Real" or metric_group in {"task", "aequitas"}:
                    out[metric_group].append({"id": id, **self._dict[id][metric_group]})
        return {
            metric_group: pd.DataFrame(metric_group_dict)
            if metric_group not in {"columnwise", "pairwise"}
            else metric_group_dict
            for metric_group, metric_group_dict in out.items()
        }

    def get_experiments(self) -> pd.DataFrame:
        """
        Return a dataframe of experiment configurations
        """
        assert hasattr(
            self, "_experiments"
        ), "You must first run `evaluate` on a `real_dataset` and set of `experiments`."
        return self._experiments

    def get_evaluation_bundle(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Return a tuple of (evaluations, experiments)
        """
        return EvalBundle(self.get_evaluations(), self.get_experiments())

    def _update(self, eval_dict, id: str) -> None:
        self._dict[id].update(eval_dict)

    def _task_step(self, data: pd.DataFrame) -> dict[str, dict]:
        metric_dict = {metric_group: {} for metric_group in self.metric_groups}
        for task in tqdm(self.tasks, desc="Running downstream tasks", leave=False):
            task_pred_column, task_metric_values = task.run(data)
            metric_dict["task"].update(task_metric_values)
            if self.aequitas and task.supports_aequitas:
                metric_dict["aequitas"].update(run_aequitas(data[self.aequitas_attributes].join(task_pred_column)))
        return metric_dict

    def _compute_metric(
        self, metric_dict: dict, metric: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        with pd.option_context("mode.chained_assignment", None), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ConvergenceWarning")
            if metric in TABLE_METRICS:
                metric_dict["table"][metric] = TABLE_METRICS[metric].compute(
                    real_data, synthetic_data, self.sdv_metadata
                )
                if issubclass(TABLE_METRICS[metric], MultiSingleColumnMetric):
                    metric_dict["columnwise"][metric] = TABLE_METRICS[metric].compute_breakdown(
                        real_data, synthetic_data, self.sdv_metadata
                    )
                elif issubclass(TABLE_METRICS[metric], MultiColumnPairsMetric):
                    metric_dict["pairwise"][metric] = TABLE_METRICS[metric].compute_breakdown(
                        real_data, synthetic_data, self.sdv_metadata
                    )
            elif metric in NUMERICAL_PRIVACY_METRICS:
                metric_dict["privacy"][metric] = NUMERICAL_PRIVACY_METRICS[metric].compute(
                    real_data.dropna(),
                    synthetic_data.dropna(),
                    self.sdv_metadata,
                    self.key_numerical_fields,
                    self.sensitive_numerical_fields,
                )
            elif metric in CATEGORICAL_PRIVACY_METRICS:
                metric_dict["privacy"][metric] = CATEGORICAL_PRIVACY_METRICS[metric].compute(
                    real_data.dropna(),
                    synthetic_data.dropna(),
                    self.sdv_metadata,
                    self.key_categorical_fields,
                    self.sensitive_categorical_fields,
                )
        return metric_dict

    def _step(self, real_data: pd.DataFrame, id: str, synthetic_data: pd.DataFrame = None) -> None:
        if synthetic_data is None:
            metric_dict = self._task_step(real_data)
        else:
            metric_dict = self._task_step(synthetic_data)
            for metric in tqdm(self.metrics, desc="Running metrics", leave=False):
                metric_dict = self._compute_metric(metric_dict, metric, real_data, synthetic_data)
        self._update(metric_dict, id)


class EvalBundle:
    def __init__(self, evaluations: pd.DataFrame, experiments: pd.DataFrame):
        self.evaluations = evaluations
        self.experiments = experiments


def validate_metric_args(
    args: argparse.Namespace, fn_dataset: str, columns: pd.Index
) -> tuple[list[Task], argparse.Namespace]:
    """
    Validate the arguments for downstream tasks
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
