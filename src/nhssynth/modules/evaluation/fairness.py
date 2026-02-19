"""
Custom fairness metrics for evaluating synthetic data.

This module provides custom implementations of fairness metrics (demographic parity,
equalized odds) for comparing real and synthetic datasets.

Metrics implemented:
    - Demographic Parity: Measures whether positive prediction rates are equal across groups.
      Lower values indicate better fairness (less disparity between groups).
    - Equalized Odds: Measures whether TPR and FPR are equal across groups.
      Lower values indicate better fairness.

Usage:
    These metrics are computed automatically when running evaluation with the --fairness flag
    and specifying protected attributes via --protected-attributes.

Example:
    nhssynth evaluate --fairness --protected-attributes age_group gender --downstream-tasks
"""

from typing import Optional

import numpy as np
import pandas as pd


def _binarize_predictions(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Convert probability predictions to binary predictions.

    Args:
        predictions: DataFrame containing prediction probabilities.
        threshold: Classification threshold.

    Returns:
        Binary predictions as a Series.
    """
    # Get the first (and typically only) column of predictions
    prob_col = predictions.iloc[:, 0]
    return (prob_col >= threshold).astype(int)


def _compute_group_rates(
    labels: pd.Series,
    predictions: pd.Series,
    group_mask: pd.Series,
) -> dict[str, float]:
    """
    Compute classification rates for a specific group.

    Args:
        labels: True labels (binary).
        predictions: Predicted labels (binary).
        group_mask: Boolean mask for the group.

    Returns:
        Dictionary with positive_rate, tpr, fpr for the group.
    """
    group_labels = labels[group_mask]
    group_preds = predictions[group_mask]

    n_group = len(group_labels)
    if n_group == 0:
        return {"positive_rate": np.nan, "tpr": np.nan, "fpr": np.nan}

    # Positive prediction rate (for demographic parity)
    positive_rate = group_preds.mean() if n_group > 0 else np.nan

    # True Positive Rate (TPR) - for equalized odds
    positives = group_labels == 1
    n_positives = positives.sum()
    tpr = group_preds[positives].mean() if n_positives > 0 else np.nan

    # False Positive Rate (FPR) - for equalized odds
    negatives = group_labels == 0
    n_negatives = negatives.sum()
    fpr = group_preds[negatives].mean() if n_negatives > 0 else np.nan

    return {"positive_rate": positive_rate, "tpr": tpr, "fpr": fpr}


def compute_demographic_parity(
    data: pd.DataFrame,
    predictions: pd.Series,
    protected_attribute: str,
) -> dict[str, float]:
    """
    Compute demographic parity metrics for a protected attribute.

    Demographic parity is satisfied when P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b.
    We report the maximum difference in positive prediction rates between any two groups.

    Args:
        data: DataFrame containing the protected attribute.
        predictions: Binary predictions.
        protected_attribute: Name of the protected attribute column.

    Returns:
        Dictionary with demographic parity metrics.
    """
    if protected_attribute not in data.columns:
        return {f"dp_{protected_attribute}_max_diff": np.nan}

    groups = data[protected_attribute].dropna().unique()
    if len(groups) < 2:
        return {f"dp_{protected_attribute}_max_diff": np.nan}

    # Align indices
    common_idx = data.index.intersection(predictions.index)
    data_aligned = data.loc[common_idx]
    preds_aligned = predictions.loc[common_idx]

    positive_rates = {}
    for group in groups:
        group_mask = data_aligned[protected_attribute] == group
        if group_mask.sum() > 0:
            positive_rates[group] = preds_aligned[group_mask].mean()

    if len(positive_rates) < 2:
        return {f"dp_{protected_attribute}_max_diff": np.nan}

    rates = list(positive_rates.values())
    max_diff = max(rates) - min(rates)

    return {f"dp_{protected_attribute}_max_diff": max_diff}


def compute_equalized_odds(
    data: pd.DataFrame,
    predictions: pd.Series,
    labels: pd.Series,
    protected_attribute: str,
) -> dict[str, float]:
    """
    Compute equalized odds metrics for a protected attribute.

    Equalized odds is satisfied when:
    - P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) (equal TPR across groups)
    - P(Ŷ=1|Y=0,A=a) = P(Ŷ=1|Y=0,A=b) (equal FPR across groups)

    We report the maximum difference in TPR and FPR between any two groups.

    Args:
        data: DataFrame containing the protected attribute.
        predictions: Binary predictions.
        labels: True binary labels.
        protected_attribute: Name of the protected attribute column.

    Returns:
        Dictionary with equalized odds metrics.
    """
    if protected_attribute not in data.columns:
        return {
            f"eo_{protected_attribute}_tpr_diff": np.nan,
            f"eo_{protected_attribute}_fpr_diff": np.nan,
        }

    groups = data[protected_attribute].dropna().unique()
    if len(groups) < 2:
        return {
            f"eo_{protected_attribute}_tpr_diff": np.nan,
            f"eo_{protected_attribute}_fpr_diff": np.nan,
        }

    # Align indices
    common_idx = data.index.intersection(predictions.index).intersection(labels.index)
    data_aligned = data.loc[common_idx]
    preds_aligned = predictions.loc[common_idx]
    labels_aligned = labels.loc[common_idx]

    tprs = {}
    fprs = {}

    for group in groups:
        group_mask = data_aligned[protected_attribute] == group
        rates = _compute_group_rates(labels_aligned, preds_aligned, group_mask)
        if not np.isnan(rates["tpr"]):
            tprs[group] = rates["tpr"]
        if not np.isnan(rates["fpr"]):
            fprs[group] = rates["fpr"]

    tpr_diff = max(tprs.values()) - min(tprs.values()) if len(tprs) >= 2 else np.nan
    fpr_diff = max(fprs.values()) - min(fprs.values()) if len(fprs) >= 2 else np.nan

    return {
        f"eo_{protected_attribute}_tpr_diff": tpr_diff,
        f"eo_{protected_attribute}_fpr_diff": fpr_diff,
    }


def run_fairness_metrics(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    protected_attributes: list[str],
    target_column: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute fairness metrics for a dataset given predictions and protected attributes.

    This function computes demographic parity and equalized odds for each protected
    attribute specified. Lower values indicate better fairness (less disparity between groups).

    Args:
        data: The full dataset containing protected attributes and target column.
        predictions: DataFrame containing prediction probabilities from the downstream task.
        protected_attributes: List of column names to use as protected attributes.
        target_column: Name of the target column (actual labels).
        threshold: Classification threshold for binarizing predictions.

    Returns:
        Dictionary mapping metric names to values. Lower values indicate better fairness.
        Metrics returned:
            - dp_{attr}_max_diff: Demographic parity - max difference in positive rates
            - eo_{attr}_tpr_diff: Equalized odds - max difference in TPR
            - eo_{attr}_fpr_diff: Equalized odds - max difference in FPR
    """
    if predictions.empty or len(predictions) == 0:
        # Task returned empty predictions (e.g., insufficient data)
        return {}

    # Binarize predictions
    binary_preds = _binarize_predictions(predictions, threshold)

    # Get labels
    if target_column not in data.columns:
        return {"fairness_error": "Target column not found"}

    labels = data[target_column]

    # Ensure indices align (predictions may be on test set only)
    # If predictions have different indices, we need to filter data to match
    if not predictions.index.equals(data.index):
        # Predictions are likely from train_test_split and have integer indices
        # We'll compute fairness on whatever subset we can align
        if len(predictions) < len(data):
            # Predictions are on a subset - use the prediction indices
            # This is tricky because the task may have reset indices
            # For now, compute on the available data
            pass

    metrics = {}

    # Filter protected attributes to those actually in the data
    valid_attrs = [attr for attr in protected_attributes if attr in data.columns and attr != target_column]

    for attr in valid_attrs:
        # Demographic parity
        dp_metrics = compute_demographic_parity(data, binary_preds, attr)
        metrics.update(dp_metrics)

        # Equalized odds
        eo_metrics = compute_equalized_odds(data, binary_preds, labels, attr)
        metrics.update(eo_metrics)

    return metrics
