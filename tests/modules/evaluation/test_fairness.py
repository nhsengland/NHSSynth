"""Tests for the fairness metrics module."""

import numpy as np
import pandas as pd
import pytest

from nhssynth.modules.evaluation.fairness import (
    _binarize_predictions,
    _compute_group_rates,
    compute_demographic_parity,
    compute_equalized_odds,
    run_fairness_metrics,
)


class TestBinarizePredictions:
    """Tests for _binarize_predictions function."""

    def test_basic_binarization(self):
        """Test basic probability to binary conversion."""
        predictions = pd.DataFrame({"prob": [0.3, 0.5, 0.7, 0.9]})
        result = _binarize_predictions(predictions, threshold=0.5)
        expected = pd.Series([0, 1, 1, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_custom_threshold(self):
        """Test binarization with custom threshold."""
        predictions = pd.DataFrame({"prob": [0.3, 0.5, 0.7, 0.9]})
        result = _binarize_predictions(predictions, threshold=0.8)
        expected = pd.Series([0, 0, 0, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_edge_cases(self):
        """Test edge case values at threshold boundary."""
        predictions = pd.DataFrame({"prob": [0.0, 0.5, 1.0]})
        result = _binarize_predictions(predictions, threshold=0.5)
        expected = pd.Series([0, 1, 1])
        pd.testing.assert_series_equal(result, expected)


class TestComputeGroupRates:
    """Tests for _compute_group_rates function."""

    def test_perfect_classifier(self):
        """Test rates for a perfect classifier."""
        labels = pd.Series([0, 0, 1, 1])
        predictions = pd.Series([0, 0, 1, 1])
        group_mask = pd.Series([True, True, True, True])

        result = _compute_group_rates(labels, predictions, group_mask)

        assert result["positive_rate"] == 0.5
        assert result["tpr"] == 1.0
        assert result["fpr"] == 0.0

    def test_always_positive_classifier(self):
        """Test rates when classifier always predicts positive."""
        labels = pd.Series([0, 0, 1, 1])
        predictions = pd.Series([1, 1, 1, 1])
        group_mask = pd.Series([True, True, True, True])

        result = _compute_group_rates(labels, predictions, group_mask)

        assert result["positive_rate"] == 1.0
        assert result["tpr"] == 1.0
        assert result["fpr"] == 1.0

    def test_empty_group(self):
        """Test rates for an empty group."""
        labels = pd.Series([0, 0, 1, 1])
        predictions = pd.Series([0, 0, 1, 1])
        group_mask = pd.Series([False, False, False, False])

        result = _compute_group_rates(labels, predictions, group_mask)

        assert np.isnan(result["positive_rate"])
        assert np.isnan(result["tpr"])
        assert np.isnan(result["fpr"])

    def test_no_positives_in_group(self):
        """Test rates when group has no positive labels."""
        labels = pd.Series([0, 0, 0, 0])
        predictions = pd.Series([0, 1, 0, 1])
        group_mask = pd.Series([True, True, True, True])

        result = _compute_group_rates(labels, predictions, group_mask)

        assert result["positive_rate"] == 0.5
        assert np.isnan(result["tpr"])  # No positives to calculate TPR
        assert result["fpr"] == 0.5

    def test_no_negatives_in_group(self):
        """Test rates when group has no negative labels."""
        labels = pd.Series([1, 1, 1, 1])
        predictions = pd.Series([0, 1, 0, 1])
        group_mask = pd.Series([True, True, True, True])

        result = _compute_group_rates(labels, predictions, group_mask)

        assert result["positive_rate"] == 0.5
        assert result["tpr"] == 0.5
        assert np.isnan(result["fpr"])  # No negatives to calculate FPR


class TestComputeDemographicParity:
    """Tests for compute_demographic_parity function."""

    def test_perfect_parity(self):
        """Test when demographic parity is perfect (no disparity)."""
        data = pd.DataFrame({"group": ["A", "A", "B", "B"]})
        predictions = pd.Series([1, 0, 1, 0])

        result = compute_demographic_parity(data, predictions, "group")

        assert result["dp_group_max_diff"] == 0.0

    def test_maximal_disparity(self):
        """Test when demographic parity has maximum disparity."""
        data = pd.DataFrame({"group": ["A", "A", "B", "B"]})
        predictions = pd.Series([1, 1, 0, 0])

        result = compute_demographic_parity(data, predictions, "group")

        assert result["dp_group_max_diff"] == 1.0

    def test_missing_attribute(self):
        """Test when protected attribute is not in data."""
        data = pd.DataFrame({"other": ["A", "A", "B", "B"]})
        predictions = pd.Series([1, 0, 1, 0])

        result = compute_demographic_parity(data, predictions, "group")

        assert np.isnan(result["dp_group_max_diff"])

    def test_single_group(self):
        """Test when there's only one group."""
        data = pd.DataFrame({"group": ["A", "A", "A", "A"]})
        predictions = pd.Series([1, 0, 1, 0])

        result = compute_demographic_parity(data, predictions, "group")

        assert np.isnan(result["dp_group_max_diff"])

    def test_nan_in_groups(self):
        """Test handling of NaN values in group column."""
        data = pd.DataFrame({"group": ["A", "A", np.nan, "B"]})
        predictions = pd.Series([1, 0, 1, 0])

        result = compute_demographic_parity(data, predictions, "group")

        # Should compute on non-NaN groups only
        assert "dp_group_max_diff" in result


class TestComputeEqualizedOdds:
    """Tests for compute_equalized_odds function."""

    def test_perfect_equality(self):
        """Test when equalized odds are perfect."""
        data = pd.DataFrame({"group": ["A", "A", "A", "A", "B", "B", "B", "B"]})
        labels = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        predictions = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])  # Perfect classifier

        result = compute_equalized_odds(data, predictions, labels, "group")

        assert result["eo_group_tpr_diff"] == 0.0
        assert result["eo_group_fpr_diff"] == 0.0

    def test_different_tpr(self):
        """Test when groups have different TPRs."""
        data = pd.DataFrame({"group": ["A", "A", "A", "A", "B", "B", "B", "B"]})
        labels = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        # Group A: TPR = 1.0 (predicts all positives correctly)
        # Group B: TPR = 0.0 (predicts all positives incorrectly)
        predictions = pd.Series([0, 0, 1, 1, 0, 0, 0, 0])

        result = compute_equalized_odds(data, predictions, labels, "group")

        assert result["eo_group_tpr_diff"] == 1.0

    def test_missing_attribute(self):
        """Test when protected attribute is not in data."""
        data = pd.DataFrame({"other": ["A", "A", "B", "B"]})
        labels = pd.Series([0, 0, 1, 1])
        predictions = pd.Series([0, 1, 0, 1])

        result = compute_equalized_odds(data, predictions, labels, "group")

        assert np.isnan(result["eo_group_tpr_diff"])
        assert np.isnan(result["eo_group_fpr_diff"])


class TestRunFairnessMetrics:
    """Tests for run_fairness_metrics function."""

    def test_basic_usage(self):
        """Test basic usage with valid inputs."""
        data = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": [0.3, 0.7, 0.3, 0.7]})

        result = run_fairness_metrics(
            data, predictions, ["group"], "target", threshold=0.5
        )

        assert "dp_group_max_diff" in result
        assert "eo_group_tpr_diff" in result
        assert "eo_group_fpr_diff" in result

    def test_empty_predictions(self):
        """Test with empty predictions DataFrame."""
        data = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": []})

        result = run_fairness_metrics(
            data, predictions, ["group"], "target"
        )

        assert result == {}

    def test_missing_target_column(self):
        """Test when target column is not in data."""
        data = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "other": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": [0.3, 0.7, 0.3, 0.7]})

        result = run_fairness_metrics(
            data, predictions, ["group"], "target"
        )

        assert "fairness_error" in result

    def test_multiple_protected_attributes(self):
        """Test with multiple protected attributes."""
        data = pd.DataFrame({
            "group1": ["A", "A", "B", "B"],
            "group2": ["X", "Y", "X", "Y"],
            "target": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": [0.3, 0.7, 0.3, 0.7]})

        result = run_fairness_metrics(
            data, predictions, ["group1", "group2"], "target"
        )

        # Should have metrics for both groups
        assert "dp_group1_max_diff" in result
        assert "dp_group2_max_diff" in result

    def test_protected_attribute_is_target(self):
        """Test that target column is excluded from protected attributes."""
        data = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": [0.3, 0.7, 0.3, 0.7]})

        result = run_fairness_metrics(
            data, predictions, ["group", "target"], "target"
        )

        # Target should be excluded from protected attributes
        assert "dp_group_max_diff" in result
        assert "dp_target_max_diff" not in result

    def test_invalid_protected_attribute(self):
        """Test with protected attribute not in data."""
        data = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "target": [0, 1, 0, 1],
        })
        predictions = pd.DataFrame({"prob": [0.3, 0.7, 0.3, 0.7]})

        result = run_fairness_metrics(
            data, predictions, ["nonexistent"], "target"
        )

        # Should return empty metrics (no valid attributes)
        assert "dp_nonexistent_max_diff" not in result
