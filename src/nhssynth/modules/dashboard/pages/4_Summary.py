"""
Summary page for NHSSynth Evaluation Dashboard.

Provides an overall summary of synthetic data quality with optional LLM-generated insights.
"""

import os
from typing import Optional

import pandas as pd
import streamlit as st

from nhssynth.modules.dashboard.utils import hide_streamlit_content

# Thresholds for rating metrics (higher is better for these metrics)
METRIC_THRESHOLDS = {
    "excellent": 0.95,
    "good": 0.85,
    "fair": 0.70,
}

# Thresholds for fairness metrics (lower is better - less disparity)
FAIRNESS_THRESHOLDS = {
    "excellent": 0.05,  # <5% difference between groups
    "good": 0.10,  # <10% difference
    "fair": 0.20,  # <20% difference
}


def get_rating(value: float, higher_is_better: bool = True) -> tuple[str, str]:
    """
    Get a rating and color for a metric value.

    Args:
        value: The metric value (typically 0-1).
        higher_is_better: Whether higher values are better.

    Returns:
        Tuple of (rating string, color).
    """
    if pd.isna(value):
        return "N/A", "gray"

    if not higher_is_better:
        value = 1 - value

    if value >= METRIC_THRESHOLDS["excellent"]:
        return "Excellent", "green"
    elif value >= METRIC_THRESHOLDS["good"]:
        return "Good", "blue"
    elif value >= METRIC_THRESHOLDS["fair"]:
        return "Fair", "orange"
    else:
        return "Poor", "red"


def get_fairness_rating(value: float) -> tuple[str, str]:
    """
    Get a rating and color for a fairness metric value.
    For fairness metrics, LOWER values are better (less disparity between groups).

    Args:
        value: The fairness metric value (difference between groups, 0-1).

    Returns:
        Tuple of (rating string, color).
    """
    if pd.isna(value):
        return "N/A", "gray"

    if value <= FAIRNESS_THRESHOLDS["excellent"]:
        return "Excellent", "green"
    elif value <= FAIRNESS_THRESHOLDS["good"]:
        return "Good", "blue"
    elif value <= FAIRNESS_THRESHOLDS["fair"]:
        return "Fair", "orange"
    else:
        return "Poor", "red"


def compute_summary_metrics(evaluations: dict, experiments: pd.DataFrame) -> dict:
    """
    Compute summary statistics from evaluations.

    Args:
        evaluations: Dictionary of evaluation DataFrames by metric group.
        experiments: DataFrame of experiment configurations.

    Returns:
        Dictionary containing summary metrics.
    """
    summary = {
        "table_metrics": {},
        "task_metrics": {},
        "privacy_metrics": {},
        "fairness_metrics": {},
        "columnwise_summary": {},
    }

    # Table-level fidelity metrics
    if "table" in evaluations:
        table_evals = evaluations["table"]
        for col in table_evals.columns:
            try:
                numeric_vals = pd.to_numeric(table_evals[col], errors="coerce")
                if not numeric_vals.isna().all():
                    mean_val = numeric_vals.mean()
                    summary["table_metrics"][col] = {
                        "mean": mean_val,
                        "min": numeric_vals.min(),
                        "max": numeric_vals.max(),
                        "rating": get_rating(mean_val),
                    }
            except Exception:
                continue

    # Task/downstream utility metrics
    if "task" in evaluations:
        task_evals = evaluations["task"]
        for col in task_evals.columns:
            try:
                numeric_vals = pd.to_numeric(task_evals[col], errors="coerce")
                if not numeric_vals.isna().all():
                    mean_val = numeric_vals.mean()
                    summary["task_metrics"][col] = {
                        "mean": mean_val,
                        "min": numeric_vals.min(),
                        "max": numeric_vals.max(),
                        "rating": get_rating(mean_val),
                    }
            except Exception:
                continue

    # Privacy metrics - need to handle non-numeric values from failed computations
    if "privacy" in evaluations:
        privacy_evals = evaluations["privacy"]
        for col in privacy_evals.columns:
            try:
                # Convert to numeric, coercing errors to NaN
                numeric_vals = pd.to_numeric(privacy_evals[col], errors="coerce")
                if not numeric_vals.isna().all():
                    mean_val = numeric_vals.mean()
                    # For privacy, higher distance/lower risk is better
                    summary["privacy_metrics"][col] = {
                        "mean": mean_val,
                        "min": numeric_vals.min(),
                        "max": numeric_vals.max(),
                        "rating": get_rating(mean_val),
                    }
            except Exception:
                # Skip metrics that can't be processed
                continue

    # Fairness metrics
    if "fairness" in evaluations:
        fairness_evals = evaluations["fairness"]
        for col in fairness_evals.columns:
            try:
                # Convert to numeric first
                numeric_col = pd.to_numeric(fairness_evals[col], errors="coerce")
                if numeric_col.isna().all():
                    continue

                # Separate real vs synthetic values for comparison
                # Handle both MultiIndex and regular Index
                if hasattr(fairness_evals.index, "get_level_values"):
                    # MultiIndex - get first level
                    idx_values = fairness_evals.index.get_level_values(0)
                    real_idx = fairness_evals.index[idx_values == "Real"]
                    synth_idx = fairness_evals.index[idx_values != "Real"]
                else:
                    # Regular index - try tuple access
                    real_idx = [
                        i for i in fairness_evals.index if (isinstance(i, tuple) and i[0] == "Real") or i == "Real"
                    ]
                    synth_idx = [
                        i
                        for i in fairness_evals.index
                        if not ((isinstance(i, tuple) and i[0] == "Real") or i == "Real")
                    ]

                real_vals = (
                    pd.to_numeric(fairness_evals.loc[real_idx, col], errors="coerce") if len(real_idx) > 0 else None
                )
                synth_vals = (
                    pd.to_numeric(fairness_evals.loc[synth_idx, col], errors="coerce") if len(synth_idx) > 0 else None
                )

                real_val = (
                    real_vals.mean()
                    if real_vals is not None and len(real_vals) > 0 and not real_vals.isna().all()
                    else None
                )
                synth_val = (
                    synth_vals.mean()
                    if synth_vals is not None and len(synth_vals) > 0 and not synth_vals.isna().all()
                    else None
                )

                # For fairness, lower is better (less disparity)
                summary["fairness_metrics"][col] = {
                    "real": real_val,
                    "synthetic_mean": synth_val,
                    "synthetic_min": (
                        synth_vals.min()
                        if synth_vals is not None and len(synth_vals) > 0 and not synth_vals.isna().all()
                        else None
                    ),
                    "synthetic_max": (
                        synth_vals.max()
                        if synth_vals is not None and len(synth_vals) > 0 and not synth_vals.isna().all()
                        else None
                    ),
                    "real_rating": get_fairness_rating(real_val) if real_val is not None else ("N/A", "gray"),
                    "synthetic_rating": get_fairness_rating(synth_val) if synth_val is not None else ("N/A", "gray"),
                }
            except Exception as e:
                # Store error for debugging
                summary["_fairness_errors"] = summary.get("_fairness_errors", [])
                summary["_fairness_errors"].append(f"{col}: {str(e)}")
                continue

    return summary


def generate_static_summary(summary: dict) -> str:
    """
    Generate a static text summary without LLM.

    Args:
        summary: Dictionary of computed summary metrics.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("## Evaluation Summary\n")

    # Table-level fidelity
    if summary["table_metrics"]:
        lines.append("### Table-Level Fidelity\n")
        lines.append("| Metric | Mean Score | Rating |")
        lines.append("|--------|------------|--------|")
        for metric, data in summary["table_metrics"].items():
            rating, color = data["rating"]
            lines.append(f"| {metric} | {data['mean']:.3f} | :{color}[{rating}] |")
        lines.append("")

    # Downstream task utility
    if summary["task_metrics"]:
        lines.append("### Downstream Task Utility\n")
        lines.append("| Metric | Mean Score | Rating |")
        lines.append("|--------|------------|--------|")
        for metric, data in summary["task_metrics"].items():
            rating, color = data["rating"]
            lines.append(f"| {metric} | {data['mean']:.3f} | :{color}[{rating}] |")
        lines.append("")

    # Privacy metrics
    if summary["privacy_metrics"]:
        lines.append("### Privacy Metrics\n")
        lines.append("| Metric | Mean Score | Rating |")
        lines.append("|--------|------------|--------|")
        for metric, data in summary["privacy_metrics"].items():
            rating, color = data["rating"]
            lines.append(f"| {metric} | {data['mean']:.3f} | :{color}[{rating}] |")
        lines.append("")

    # Fairness metrics
    if summary["fairness_metrics"]:
        lines.append("### Fairness Metrics\n")
        lines.append("*Lower values indicate better fairness (less disparity between groups)*\n")
        lines.append("| Metric | Real Data | Synthetic Data | Fairness Preserved? |")
        lines.append("|--------|-----------|----------------|---------------------|")
        for metric, data in summary["fairness_metrics"].items():
            real_val = f"{data['real']:.3f}" if data["real"] is not None else "N/A"
            synth_val = f"{data['synthetic_mean']:.3f}" if data["synthetic_mean"] is not None else "N/A"
            real_rating, real_color = data["real_rating"]
            synth_rating, synth_color = data["synthetic_rating"]

            # Check if fairness is preserved (synthetic is similar or better)
            if data["real"] is not None and data["synthetic_mean"] is not None:
                diff = abs(data["synthetic_mean"] - data["real"])
                if diff <= 0.05:
                    preserved = ":green[Yes]"
                elif data["synthetic_mean"] < data["real"]:
                    preserved = ":green[Improved]"
                elif diff <= 0.10:
                    preserved = ":orange[Mostly]"
                else:
                    preserved = ":red[No]"
            else:
                preserved = "N/A"

            lines.append(
                f"| {metric} | {real_val} (:{real_color}[{real_rating}]) | {synth_val} (:{synth_color}[{synth_rating}]) | {preserved} |"
            )
        lines.append("")

    # Interpretation
    lines.append("### Interpretation\n")

    # Overall fidelity assessment
    if summary["table_metrics"]:
        fidelity_scores = [d["mean"] for d in summary["table_metrics"].values() if not pd.isna(d["mean"])]
        if fidelity_scores:
            avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
            rating, _ = get_rating(avg_fidelity)
            lines.append(f"**Overall Fidelity**: {rating} (average score: {avg_fidelity:.3f})")
            if avg_fidelity >= 0.90:
                lines.append("- The synthetic data closely matches the statistical properties of the real data.")
            elif avg_fidelity >= 0.80:
                lines.append("- The synthetic data reasonably captures the main patterns in the real data.")
            else:
                lines.append("- There are notable differences between synthetic and real data distributions.")

    # Task utility assessment
    if summary["task_metrics"]:
        task_scores = [d["mean"] for d in summary["task_metrics"].values() if not pd.isna(d["mean"])]
        if task_scores:
            avg_task = sum(task_scores) / len(task_scores)
            rating, _ = get_rating(avg_task)
            lines.append(f"\n**Downstream Utility**: {rating} (average score: {avg_task:.3f})")
            if avg_task >= 0.85:
                lines.append("- Models trained on synthetic data perform comparably to those trained on real data.")
            elif avg_task >= 0.70:
                lines.append("- Moderate utility for downstream tasks; some performance degradation expected.")
            else:
                lines.append("- Limited utility for downstream ML tasks; consider tuning generation parameters.")

    # Fairness assessment
    if summary["fairness_metrics"]:
        # Check how many metrics preserved fairness
        preserved_count = 0
        total_count = 0
        for data in summary["fairness_metrics"].values():
            if data["real"] is not None and data["synthetic_mean"] is not None:
                total_count += 1
                diff = abs(data["synthetic_mean"] - data["real"])
                if diff <= 0.10 or data["synthetic_mean"] < data["real"]:
                    preserved_count += 1

        if total_count > 0:
            preservation_rate = preserved_count / total_count
            lines.append(f"\n**Fairness Preservation**: {preserved_count}/{total_count} metrics maintained")
            if preservation_rate >= 0.8:
                lines.append("- Synthetic data maintains similar fairness properties to the original data.")
            elif preservation_rate >= 0.5:
                lines.append("- Some fairness properties are preserved, but others show increased disparity.")
            else:
                lines.append("- Fairness properties differ significantly; review protected attribute handling.")

    return "\n".join(lines)


def generate_llm_summary(summary: dict, api_key: str, model: str = "claude-3-haiku-20240307") -> Optional[str]:
    """
    Generate an LLM-powered summary of evaluation results.

    Args:
        summary: Dictionary of computed summary metrics.
        api_key: Anthropic API key.
        model: Model to use for generation.

    Returns:
        LLM-generated summary string, or None if generation fails.
    """
    try:
        import anthropic
    except ImportError:
        st.warning(
            "The `anthropic` package is not installed. Install it with `pip install anthropic` to enable LLM summaries."
        )
        return None

    # Build the prompt with evaluation data
    prompt_parts = ["Analyze these synthetic data evaluation metrics and provide a concise summary:\n"]

    if summary["table_metrics"]:
        prompt_parts.append("\nTable-Level Fidelity Metrics:")
        for metric, data in summary["table_metrics"].items():
            if not pd.isna(data["mean"]):
                prompt_parts.append(f"- {metric}: {data['mean']:.3f} (range: {data['min']:.3f}-{data['max']:.3f})")

    if summary["task_metrics"]:
        prompt_parts.append("\nDownstream Task Utility Metrics:")
        for metric, data in summary["task_metrics"].items():
            if not pd.isna(data["mean"]):
                prompt_parts.append(f"- {metric}: {data['mean']:.3f} (range: {data['min']:.3f}-{data['max']:.3f})")

    if summary["privacy_metrics"]:
        prompt_parts.append("\nPrivacy Metrics:")
        for metric, data in summary["privacy_metrics"].items():
            if not pd.isna(data["mean"]):
                prompt_parts.append(f"- {metric}: {data['mean']:.3f} (range: {data['min']:.3f}-{data['max']:.3f})")

    if summary["fairness_metrics"]:
        prompt_parts.append("\nFairness Metrics (lower is better - less disparity between groups):")
        for metric, data in summary["fairness_metrics"].items():
            real_str = f"{data['real']:.3f}" if data["real"] is not None else "N/A"
            synth_str = f"{data['synthetic_mean']:.3f}" if data["synthetic_mean"] is not None else "N/A"
            prompt_parts.append(f"- {metric}: Real={real_str}, Synthetic={synth_str}")

    prompt_parts.append("\nProvide a brief summary (2-3 paragraphs) that:")
    prompt_parts.append("1. Assesses overall synthetic data quality")
    prompt_parts.append("2. Highlights strengths and areas for improvement")
    prompt_parts.append("3. Provides actionable recommendations if applicable")
    prompt_parts.append("\nUse clear, non-technical language where possible.")

    prompt = "\n".join(prompt_parts)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
        return message.content[0].text
    except Exception as e:
        st.error(f"Error generating LLM summary: {e}")
        return None


def page():
    st.write("# Evaluation Summary")
    st.write("An overview of synthetic data quality based on evaluation metrics.")

    evaluations = st.session_state["evaluations"]
    experiments = st.session_state["experiments"]

    # Sidebar configuration
    st.sidebar.write("### Summary Options")

    # LLM configuration
    enable_llm = st.sidebar.checkbox(
        "Enable LLM Summary",
        value=False,
        help="Use an LLM to generate natural language insights about the evaluation results.",
    )

    api_key = None
    if enable_llm:
        # Check for API key in environment first
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            st.sidebar.success("Using API key from environment.")
            api_key = env_key
        else:
            api_key = st.sidebar.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key to enable LLM-powered summaries.",
            )
            if not api_key:
                st.sidebar.warning("Enter an API key or set ANTHROPIC_API_KEY environment variable.")

    # Compute summary metrics
    summary = compute_summary_metrics(evaluations, experiments)

    # Generate and display summary
    if enable_llm and api_key:
        with st.spinner("Generating LLM summary..."):
            llm_summary = generate_llm_summary(summary, api_key)

        if llm_summary:
            st.write("## AI-Generated Insights")
            st.write(llm_summary)
            st.divider()

    # Always show static summary with metrics tables
    static_summary = generate_static_summary(summary)
    st.markdown(static_summary)

    # Additional details section
    with st.expander("View Raw Metrics"):
        if summary["table_metrics"]:
            st.write("#### Table Metrics")
            table_df = pd.DataFrame(summary["table_metrics"]).T
            table_df["rating"] = table_df["rating"].apply(lambda x: x[0])
            st.dataframe(table_df)

        if summary["task_metrics"]:
            st.write("#### Task Metrics")
            task_df = pd.DataFrame(summary["task_metrics"]).T
            task_df["rating"] = task_df["rating"].apply(lambda x: x[0])
            st.dataframe(task_df)

        if summary["privacy_metrics"]:
            st.write("#### Privacy Metrics")
            privacy_df = pd.DataFrame(summary["privacy_metrics"]).T
            privacy_df["rating"] = privacy_df["rating"].apply(lambda x: x[0])
            st.dataframe(privacy_df)

        if summary["fairness_metrics"]:
            st.write("#### Fairness Metrics")
            st.write("*Lower values indicate better fairness (less disparity between groups)*")
            fairness_df = pd.DataFrame(summary["fairness_metrics"]).T
            fairness_df["real_rating"] = fairness_df["real_rating"].apply(lambda x: x[0])
            fairness_df["synthetic_rating"] = fairness_df["synthetic_rating"].apply(lambda x: x[0])
            st.dataframe(fairness_df)

        # Debug: Show if fairness was expected but not computed
        if "fairness" in evaluations and not summary["fairness_metrics"]:
            st.write("#### Fairness Metrics")
            st.warning("Fairness evaluation data exists but no metrics could be extracted.")
            fairness_raw = evaluations["fairness"]
            st.write(f"Columns: {list(fairness_raw.columns)}")
            st.write(f"Shape: {fairness_raw.shape}")
            st.write(f"Index type: {type(fairness_raw.index)}")
            if "_fairness_errors" in summary:
                st.error(f"Errors: {summary['_fairness_errors']}")
            if not fairness_raw.empty:
                st.dataframe(fairness_raw)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    hide_streamlit_content()
    if "evaluations" not in st.session_state:
        st.error("Upload an evaluation bundle to get started!")
    else:
        page()
