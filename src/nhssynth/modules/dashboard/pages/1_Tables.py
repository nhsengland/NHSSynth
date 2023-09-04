import pandas as pd
import streamlit as st
from nhssynth.modules.dashboard.utils import hide_streamlit_content, id_selector


def convert_df(df):
    return df.to_csv().encode("utf-8")


def prepare_evals(evaluations: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    # Ensure that we can only select from metrics that have been computed for this set of evaluations
    valid_metrics = [m for m in evaluations.columns if not evaluations[m].isna().all()]
    metrics_to_show = st.sidebar.multiselect("Select metric(s) to display", valid_metrics, default=valid_metrics)
    # Filter to only the columns we want to display
    return evaluations[metrics_to_show].join(experiments)[experiments.columns.tolist() + metrics_to_show]


def group_by_display(
    evaluations_filtered: pd.DataFrame,
    experiments: pd.DataFrame,
    title: str,
):
    group_by_cols = st.sidebar.multiselect(
        "Select column(s) to group by",
        experiments.columns.tolist() + experiments.index.names,
        default=["architecture", "config"],
    )
    if group_by_cols:
        evaluations_shown = (
            evaluations_filtered[
                [
                    c
                    for c in evaluations_filtered.columns
                    if c not in experiments.columns.tolist() + experiments.index.names or c in group_by_cols
                ]
            ]
            .groupby(group_by_cols, dropna=False)
            .mean()
        )
        st.write(f"## {title} grouped by `{group_by_cols}`")
        download = convert_df(evaluations_shown.copy())
        # No idea why this fixes the bug with highlight_max but it does
        evaluations_shown.index = evaluations_shown.index.set_levels(evaluations_shown.index.levels[0], level=0)
        st.dataframe(evaluations_shown.style.highlight_max(axis=0))
        st.download_button("Press to Download", download, "evaluations.csv", "text/csv")
        if "config" in group_by_cols:
            st.write("### Configurations")
            st.dataframe(experiments.groupby(["architecture", "config"]).first())
    else:
        evaluations_shown = evaluations_filtered
        st.write(f"## {title}")
        st.dataframe(evaluations_shown.style.highlight_max(axis=0))
        st.download_button("Press to Download", convert_df(evaluations_shown), "evaluations.csv", "text/csv")


def table_metrics(evaluations, experiments, selected_metric_group):
    prepared_evals = prepare_evals(evaluations, experiments)
    group_by_display(prepared_evals, experiments, f"{selected_metric_group.capitalize()} metrics")


def columnwise_metrics(evaluations, experiments):
    display = st.sidebar.selectbox("Display", ["By column", "By metric", "By configuration"], index=0)

    if display == "By column":
        columns = evaluations.iloc[0, 0].keys()
        column = st.sidebar.selectbox("Select column to display", columns)
        evaluations = evaluations.apply(lambda x: x.apply(lambda y: y[column]["score"]))
        prepared_evals = prepare_evals(evaluations, experiments)
        group_by_display(prepared_evals, experiments, f"Columnwise metrics for `{column}`")

    elif display == "By metric":
        metric = st.sidebar.selectbox("Select metric to display", evaluations.columns)
        metric_evaluations = (
            evaluations[metric].apply(lambda x: pd.Series(x).apply(lambda y: y["score"])).dropna(axis=1, how="all")
        )
        joined_evals = metric_evaluations.join(experiments)
        group_by_display(joined_evals, experiments, f"Columnwise `{metric}`")

    elif display == "By configuration":
        config_evaluations = id_selector(evaluations)
        exploded_evaluations = config_evaluations.apply(lambda x: pd.Series(x).apply(lambda y: y["score"])).T
        metrics_to_show = st.sidebar.multiselect(
            "Select metric(s) to display",
            exploded_evaluations.columns.tolist(),
            default=exploded_evaluations.columns.tolist(),
        )
        exploded_evaluations = exploded_evaluations[metrics_to_show]
        st.write(
            f"## Columnwise metrics for {config_evaluations.name[0]} repeat {int(config_evaluations.name[1])} configuration {int(config_evaluations.name[2])}"
        )
        st.dataframe(exploded_evaluations.style.highlight_max(axis=0))
        st.download_button(
            "Press to Download", convert_df(exploded_evaluations), "evaluations.csv", "text/csv", key="download-csv"
        )
        st.write("### Configurations")
        st.dataframe(experiments.groupby(["architecture", "config"]).first())


def pairwise_metrics(evaluations, experiments):
    display = st.sidebar.selectbox(
        "Display",
        ["By column pair", "By reference column and metric", "By configuration and metric"],
        index=0,
    )
    metric_column_map = {
        metric: set([item for pair in evaluations[metric][0].keys() for item in pair]) for metric in evaluations.columns
    }
    columns = set([item for sublist in metric_column_map.values() for item in sublist])
    column_column_map = {column: set() for column in columns}
    for column in columns:
        for metric_columns in metric_column_map.values():
            if column in metric_columns:
                column_column_map[column] = column_column_map[column].union(metric_columns).difference({column})
    if display == "By column pair":
        column1 = st.sidebar.selectbox("Select first column to display", columns, index=0)
        column2 = st.sidebar.selectbox("Select second column to display", column_column_map[column1], index=0)
        column_pair_evaluations = evaluations.apply(
            lambda x: x.apply(lambda y: (y.get((column1, column2)) or y.get((column2, column1)) or {}).get("score"))
        )
        prepared_evals = prepare_evals(column_pair_evaluations, experiments)
        group_by_display(prepared_evals, experiments, f"Pairwise metrics for `{column1}` and `{column2}`")
    elif display == "By reference column and metric":
        reference_column = st.sidebar.selectbox("Select reference column", columns, index=0)
        metric = st.sidebar.selectbox(
            "Select metric to display", [k for k, v in metric_column_map.items() if reference_column in v]
        )
        metric_evaluations = (
            evaluations[metric]
            .apply(pd.Series)
            .filter(like=reference_column)
            .apply(lambda x: x.apply(lambda y: y["score"]))
        )
        metric_evaluations.columns = [
            (col[0] + col[1]).replace(reference_column, "") for col in metric_evaluations.columns
        ]
        joined_evals = metric_evaluations.join(experiments)
        group_by_display(joined_evals, experiments, f"Pairwise `{metric}` relative to `{reference_column}`")
    elif display == "By configuration and metric":
        config_evaluations = id_selector(evaluations)
        metric = st.sidebar.selectbox("Select metric to display", evaluations.columns)
        metric_evaluations = config_evaluations.loc[metric]
        grid = pd.DataFrame(columns=list(metric_column_map[metric]), index=list(metric_column_map[metric]))
        for (col1, col2), score in metric_evaluations.items():
            grid.loc[col1, col2], grid.loc[col2, col1] = score["score"], score["score"]
        st.write(
            f"## Pairwise `{metric}` values for {config_evaluations.name[0]} repeat {int(config_evaluations.name[1])} configuration {int(config_evaluations.name[2])}"
        )
        st.table(grid)
        st.download_button("Press to Download", convert_df(grid), "evaluations.csv", "text/csv", key="download-csv")
        st.write("### Configurations")
        st.dataframe(experiments.groupby(["architecture", "config"]).first())


def page():
    metric_groups = st.session_state["evaluations"].keys()
    selected_metric_group = st.sidebar.selectbox(
        "Select a metric group", [mg.capitalize() for mg in list(metric_groups)]
    ).lower()

    evaluations = st.session_state["evaluations"][selected_metric_group].copy()
    experiments = st.session_state["experiments"]
    if selected_metric_group in ["table", "task", "privacy", "efficacy", "aequitas"]:
        table_metrics(evaluations, experiments, selected_metric_group)
    elif selected_metric_group == "columnwise":
        columnwise_metrics(evaluations, experiments)
    elif selected_metric_group == "pairwise":
        pairwise_metrics(evaluations, experiments)
    else:
        st.error("Not yet implemented.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    hide_streamlit_content()
    if "evaluations" not in st.session_state:
        st.error("Upload an evaluation bundle to get started!")
    else:
        page()
