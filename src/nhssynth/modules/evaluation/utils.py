import argparse
import itertools
import warnings
from typing import Union

import pandas as pd
from nhssynth.common.constants import (
    CATEGORICAL_PRIVACY_METRICS,
    METRIC_CHOICES,
    NUMERICAL_PRIVACY_METRICS,
    TABLE_METRICS,
)
from nhssynth.modules.evaluation.aequitas import run_aequitas
from nhssynth.modules.evaluation.tasks import Task, get_tasks
from sdmetrics.single_table import MultiColumnPairsMetric, MultiSingleColumnMetric
from tqdm import tqdm


class EvalFrame:
    def __init__(
        self,
        experiment_bundle: dict[str, Union[pd.DataFrame, dict[str, pd.DataFrame]]],
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
        self.architectures = list(experiment_bundle.keys())
        if isinstance(experiment_bundle[self.architectures[0]], dict):
            self.seeds = list(experiment_bundle[self.architectures[0]].keys())
        else:
            self.seeds = None

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

        self.metric_groups = ["task", "aequitas", "columnwise", "pairwise", "table", "privacy", "efficacy"]
        self._dict = self._init_dict()

    def _init_dict(self) -> dict[str, dict]:
        return {
            **{arch: {seed: {} for seed in self.seeds} if self.seeds else {} for arch in self.architectures},
            **{"Real": {}},
        }

    def __iter__(self):
        if self.seeds:
            return itertools.product(self.architectures, self.seeds)
        else:
            return (arch for arch in self.architectures)

    def __len__(self) -> int:
        return len(self.architectures) * (len(self.seeds) if self.seeds else 1)

    def collect(self) -> dict[str, pd.DataFrame]:
        """
        Return a dict of dataframes each with one column for seed, one for architecture, and one per metric / report
        """
        out = {}
        for metric_group in self.metric_groups:
            if metric_group in ["task", "aequitas"]:
                out[metric_group] = [{"architecture": "Real", "seed": None, **self._dict["Real"].get(metric_group, {})}]
            else:
                out[metric_group] = []
            for arch in self.architectures:
                if self.seeds:
                    for seed in self.seeds:
                        out[metric_group].append(
                            {"architecture": arch, "seed": seed, **self._dict[arch][seed][metric_group]}
                        )
                else:
                    out[metric_group].append({"architecture": arch, **self._dict[arch][metric_group]})
        return {
            metric_group: pd.DataFrame(metric_group_dict)
            if metric_group not in ["columnwise", "pairwise"]
            else metric_group_dict
            for metric_group, metric_group_dict in out.items()
        }

    def get_arch(self, arch) -> dict[str, pd.DataFrame]:
        """
        Access the contents of self._dict by architecture, flipping the internal dict (which is keyed by seed) to a dict of dataframes keyed by metric group
        """
        out = {}
        for metric_group in self.metric_groups:
            out[metric_group] = {}
            for seed in self.seeds:
                out[metric_group][seed] = self._dict[arch][seed][metric_group]
            out[metric_group] = pd.DataFrame(out[metric_group])
        return out

    def get_eval(self, arch, seed) -> dict[str, float]:
        return self._dict[arch][seed]

    def _update(self, eval_dict, step: Union[str, tuple[str, int]]) -> None:
        if isinstance(step, tuple):
            arch, seed = step
            self._dict[arch][seed].update(eval_dict)
        else:
            arch = step
            self._dict[arch].update(eval_dict)

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
                if isinstance(TABLE_METRICS[metric], MultiSingleColumnMetric):
                    metric_dict["columnwise"][metric] = TABLE_METRICS[metric].compute_breakdown(
                        real_data, synthetic_data, self.sdv_metadata
                    )
                elif isinstance(TABLE_METRICS[metric], MultiColumnPairsMetric):
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

    def step(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame = None,
        step: Union[str, tuple[str, int]] = None,
    ) -> None:
        if synthetic_data is None:
            metric_dict = self._task_step(real_data)
            self._update(metric_dict, step="Real")
        else:
            metric_dict = self._task_step(synthetic_data)
            for metric in tqdm(self.metrics, desc="Running metrics", leave=False):
                metric_dict = self._compute_metric(metric_dict, metric, real_data, synthetic_data)
            self._update(metric_dict, step)


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
