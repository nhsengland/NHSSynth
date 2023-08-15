"""Single table quality report."""

import itertools
import pickle

import numpy as np
import pandas as pd
from nhssynth.common.constants import *
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.single_table.plot_utils import *
from sdmetrics.reports.utils import *
from sdmetrics.single_table import *
from tqdm import tqdm


class FullReport:
    """Single table full report."""

    def __init__(self, metrics, metric_args={}):
        self._overall_quality_score = None
        self._metric_results = {}
        self._metric_averages = {}
        self._property_breakdown = {}
        self._property_errors = {}
        self._metrics = metrics
        self._metric_args = metric_args

    def _get_metric_scores(self, metric_name):
        """Aggregate the scores and errors in a metric results mapping.

        Args:
            The metric results to aggregate.

        Returns:
            The average of the metric scores, and the number of errors.
        """
        metric_results = self._metric_results.get(metric_name, {})
        if len(metric_results) == 0:
            return np.nan
        metric_scores = []
        for breakdown in metric_results.values():
            if isinstance(breakdown, dict):
                metric_score = breakdown.get("score", np.nan)
                if not np.isnan(metric_score):
                    metric_scores.append(metric_score)
            else:
                return [metric_results.get("score", np.nan)]
        return metric_scores

    def _print_results(self):
        """Print the quality report results."""
        if pd.isna(self._overall_quality_score) & any(self._property_errors.values()):
            print("\nOverall Score: Error computing report.")
        else:
            print(f"\nOverall Score: {round(self._overall_quality_score * 100, 2)}%")

        if len(self._property_breakdown) > 0:
            print("\nProperties:")

        for prop, score in self._property_breakdown.items():
            if not pd.isna(score):
                print(f"{prop}: {round(score * 100, 2)}%")
            elif self._property_errors[prop] > 0:
                print(f"{prop}: Error computing property.")
            else:
                print(f"{prop}: NaN")

        print("")

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            verbose (bool):
                Whether or not to print report summary and progress.
        """
        print("")
        validate_single_table_inputs(real_data, synthetic_data, metadata)
        self._property_breakdown = {}
        for prop, metrics in tqdm(
            self._metrics.items(), desc="Creating report", position=0, disable=(not verbose), leave=True
        ):
            num_prop_errors = 0
            if "NewRowSynthesis" in METRIC_CHOICES[prop]:
                if "NewRowSynthesis" not in self._metric_args:
                    self._metric_args["NewRowSynthesis"] = {}
                self._metric_args["NewRowSynthesis"]["synthetic_sample_size"] = min(
                    min(len(real_data), len(synthetic_data)),
                    self._metric_args["NewRowSynthesis"].get("synthetic_sample_size", len(real_data)),
                )
            for metric in tqdm(metrics, desc=prop + " metrics", position=1, disable=(not verbose), leave=False):
                metric_name = metric.__name__
                try:
                    metric_args = self._metric_args.get(metric_name, {})
                    metric_results = metric.compute_breakdown(real_data, synthetic_data, metadata, **metric_args)
                    if "score" in metric_results:
                        metric_average = metric_results["score"]
                        num_prop_errors += metric_results.get("error", 0)
                    else:
                        metric_average, num_metric_errors = aggregate_metric_results(metric_results)
                        num_prop_errors += num_metric_errors

                except IncomputableMetricError:
                    # Metric is not compatible with this dataset.
                    metric_results = {}
                    metric_average = np.nan
                    num_prop_errors += 1

                self._metric_averages[metric_name] = metric_average
                self._metric_results[metric_name] = metric_results

            if (
                prop == "Column Similarity"
                and "ContingencySimilarity" in self._metric_results
                and "CorrelationSimilarity" in self._metric_results
            ):
                existing_column_pairs = list(self._metric_results["ContingencySimilarity"].keys())
                existing_column_pairs.extend(list(self._metric_results["CorrelationSimilarity"].keys()))
                additional_results = discretize_and_apply_metric(
                    real_data, synthetic_data, metadata, ContingencySimilarity, existing_column_pairs
                )
                self._metric_results["ContingencySimilarity"].update(additional_results)
                self._metric_averages["ContingencySimilarity"], _ = aggregate_metric_results(
                    self._metric_results["ContingencySimilarity"]
                )

            self._property_breakdown[prop] = np.mean([s for m in metrics for s in self._get_metric_scores(m.__name__)])
            self._property_errors[prop] = num_prop_errors

        self._overall_quality_score = np.nanmean(list(self._property_breakdown.values()))

        if verbose:
            self._print_results()

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        return self._overall_quality_score

    def get_properties(self):
        """Return the property score breakdown.

        Returns:
            pandas.DataFrame
                The property score breakdown.
        """
        return pd.DataFrame(
            {
                "Property": self._property_breakdown.keys(),
                "Score": self._property_breakdown.values(),
            }
        )

    def get_visualization(self, property_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        score_breakdowns = {
            metric.__name__: self._metric_results[metric.__name__] for metric in self._metrics.get(property_name, [])
        }

        if property_name == "Column Shape":
            fig = get_column_shapes_plot(score_breakdowns, self._property_breakdown[property_name])
            fig.show()

        elif property_name == "Column Similarity":
            fig = get_column_pairs_plot(
                score_breakdowns,
                self._property_breakdown[property_name],
            )
            fig.show()

        elif property_name == "Coverage":
            fig = get_column_coverage_plot(score_breakdowns, self._property_breakdown[property_name])
            fig.show()

        elif property_name == "Boundary":
            fig = get_column_boundaries_plot(score_breakdowns, self._property_breakdown[property_name])
            fig.show()

        elif property_name == "Synthesis":
            fig = get_synthesis_plot(score_breakdowns.get("NewRowSynthesis", {}))
            fig.show()

        elif property_name == "Detection":
            print("WARNING: Detection plots not currently implemented.")

        elif property_name == "Divergence":
            print("WARNING: Divergence plots not currently implemented.")

        else:
            raise ValueError(f"Property name `{property_name}` is not recognized / supported.")

    def get_details(self, property_name):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            pandas.DataFrame
                The score breakdown.
        """
        columns = []
        metrics = []
        scores = []
        errors = []
        details = pd.DataFrame()

        if property_name == "Detection":
            for metric in self._metrics[property_name]:
                metric_results = self._metric_results[metric.__name__]
                if "score" in metric_results and pd.isna(metric_results["score"]):
                    continue
                metrics.append(metric.__name__)
                scores.append(metric_results.get("score", np.nan))
                errors.append(metric_results.get("error", np.nan))

            details = pd.DataFrame(
                {
                    "Metric": metrics,
                    "Overall Score": scores,
                }
            )

        elif property_name == "Column Shape":
            for metric in self._metrics[property_name]:
                for column, score_breakdown in self._metric_results[metric.__name__].items():
                    if "score" in score_breakdown and pd.isna(score_breakdown["score"]):
                        continue

                    columns.append(column)
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown.get("score", np.nan))
                    errors.append(score_breakdown.get("error", np.nan))

            details = pd.DataFrame(
                {
                    "Column": columns,
                    "Metric": metrics,
                    "Overall Score": scores,
                }
            )

        elif property_name == "Column Similarity" or property_name == "Divergence":
            real_scores = []
            synthetic_scores = []
            for metric in self._metrics[property_name]:
                for column_pair, score_breakdown in self._metric_results[metric.__name__].items():
                    columns.append(column_pair)
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown.get("score", np.nan))
                    if property_name == "Column Similarity":
                        real_scores.append(score_breakdown.get("real", np.nan))
                        synthetic_scores.append(score_breakdown.get("synthetic", np.nan))
                    errors.append(score_breakdown.get("error", np.nan))

            details = pd.DataFrame(
                {
                    "Column 1": [col1 for col1, _ in columns],
                    "Column 2": [col2 for _, col2 in columns],
                    "Metric": metrics,
                    "Overall Score": scores,
                    "Real Correlation": real_scores,
                    "Synthetic Correlation": synthetic_scores,
                }
            )

        elif property_name == "Synthesis":
            metric_name = self._metrics[property_name][0].__name__
            metric_result = self._metric_results[metric_name]
            details = pd.DataFrame(
                {
                    "Metric": [metric_name],
                    "Overall Score": [metric_result.get("score", np.nan)],
                    "Num Matched Rows": [metric_result.get("num_matched_rows", np.nan)],
                    "Num New Rows": [metric_result.get("num_new_rows", np.nan)],
                }
            )
            errors.append(metric_result.get("error", np.nan))

        else:
            for metric in self._metrics[property_name]:
                for column, score_breakdown in self._metric_results[metric.__name__].items():
                    metric_score = score_breakdown.get("score", np.nan)
                    metric_error = score_breakdown.get("error", np.nan)
                    if pd.isna(metric_score) and pd.isna(metric_error):
                        continue

                    columns.append(column)
                    metrics.append(metric.__name__)
                    scores.append(metric_score)
                    errors.append(metric_error)

            details = pd.DataFrame(
                {
                    "Column": columns,
                    "Metric": metrics,
                    "Overall Score": scores,
                }
            )

        if pd.Series(errors).notna().sum() > 0:
            details["Error"] = errors

        return details

    def get_raw_result(self, metric_name):
        """Return the raw result of the given metric name.

        Args:
            metric_name (str):
                The name of the desired metric.

        Returns:
            dict
                The raw results
        """
        metrics = list(itertools.chain.from_iterable(self._metrics.values()))
        for metric in metrics:
            if metric.__name__ == metric_name:
                return [
                    {
                        "metric": {
                            "method": f"{metric.__module__}.{metric.__name__}",
                            "parameters": {},
                        },
                        "results": {
                            key: result
                            for key, result in self._metric_results[metric_name].items()
                            if not pd.isna(result.get("score", np.nan))
                        },
                    },
                ]

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        with open(filepath, "wb") as output:
            pickle.dump(self, output)
