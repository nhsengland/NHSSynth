import pandas as pd
import streamlit as st


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def prepare_evals(evaluations: pd.DataFrame, experiments: pd.DataFrame, metrics: list, metadata: list) -> pd.DataFrame:
    # Replace the ID column with the configuration components
    evaluations_joined = evaluations.join(experiments.set_index("id"), on="id").drop(columns=["id"])
    # Ensure that we can only select from metrics that have been computed for this set of evaluations
    valid_metrics = [m for m in metrics if not evaluations_joined[m].isna().all()]
    metrics_to_show = st.sidebar.multiselect("Select metric(s) to display", valid_metrics, default=valid_metrics)
    # Filter to only the columns we want to display
    return evaluations_joined[metadata + metrics_to_show]


def group_by_display(
    evaluations_filtered: pd.DataFrame,
    metadata: list,
    selected_metric_group: str,
    experiments: pd.DataFrame,
):
    group_by_cols = st.sidebar.multiselect(
        "Select column(s) to group by", metadata, default=["architecture", "config_idx"]
    )
    if group_by_cols:
        evaluations_shown = (
            evaluations_filtered[[c for c in evaluations_filtered.columns if c not in metadata or c in group_by_cols]]
            .replace(float("nan"), "N/A")
            .groupby(group_by_cols)
            .mean()
        )
        st.write(f"## `{selected_metric_group}` metrics grouped by `{group_by_cols}`")
        st.dataframe(evaluations_shown.style.highlight_max(axis=0))
        st.download_button(
            "Press to Download", convert_df(evaluations_shown), "evaluations.csv", "text/csv", key="download-csv"
        )
        if "config_idx" in group_by_cols:
            st.write("### Configurations")
            st.dataframe(
                experiments[metadata].drop(["seed", "repeat"], axis=1).groupby(["architecture", "config_idx"]).first()
            )
    else:
        evaluations_shown = evaluations_filtered
        st.write(f"## `{selected_metric_group}` metrics")
        st.dataframe(evaluations_shown.style.highlight_max(axis=0))
        st.download_button(
            "Press to Download", convert_df(evaluations_shown), "evaluations.csv", "text/csv", key="download-csv"
        )


def id_selector(experiments):
    architecture = st.sidebar.selectbox(
        "Select architecture to display", [a for a in experiments["architecture"].unique() if a != "Real"]
    )
    id = architecture
    config_ids = experiments[experiments["architecture"] == architecture]["config_idx"].dropna().unique()
    repeats = experiments[experiments["architecture"] == architecture]["repeat"].dropna().unique()
    if len(config_ids) > 1:
        config = st.sidebar.selectbox("Select configuration to display", config_ids, index=0)
        id += f"_config_{config}"
    if len(experiments[experiments["architecture"] == architecture]["repeat"].dropna().unique()) > 1:
        repeat = st.sidebar.selectbox("Select repeat to display", repeats, index=0)
        id += f"_repeat_{repeat}"
    return id


def table_metrics(evaluations, experiments, selected_metric_group):
    metrics = [c for c in evaluations.columns if c != "id"]
    metadata = [c for c in experiments.columns if c != "id"]
    prepared_evals = prepare_evals(evaluations, experiments, metrics, metadata)
    group_by_display(prepared_evals, metadata, selected_metric_group, experiments)


def columnwise_metrics(evaluations, experiments):
    evaluation_df = pd.DataFrame(evaluations)
    metrics = [c for c in evaluation_df.columns if c != "id"]
    metadata = [c for c in experiments.columns if c not in ["id"]]

    display = st.sidebar.selectbox("Display", ["By column", "By metric", "By configuration"], index=0)

    if display == "By column":
        columns = evaluation_df[metrics[0]][1].keys()
        column = st.sidebar.selectbox("Select column to display", columns)
        column_evaluations = evaluation_df.copy()
        for metric in metrics:
            column_evaluations[metric] = column_evaluations[metric].apply(lambda x: x[column]["score"])
        prepared_evals = prepare_evals(column_evaluations, experiments, metrics, metadata)
        group_by_display(prepared_evals, metadata, "columnwise", experiments)

    elif display == "By metric":
        metric = st.sidebar.selectbox("Select metric to display", metrics)
        metric_evaluations = pd.concat(
            [
                evaluation_df["id"],
                evaluation_df[metric].apply(lambda x: pd.Series({k: v["score"] for k, v in x.items()})),
            ],
            axis=1,
        )
        prepared_evals = metric_evaluations.join(experiments.set_index("id"), on="id").drop(
            columns=["id"] + metric_evaluations.columns[metric_evaluations.isnull().all()].tolist()
        )
        group_by_display(prepared_evals, metadata, "columnwise", experiments)

    elif display == "By configuration":
        id = id_selector(experiments)
        config_evaluations = evaluation_df[evaluation_df["id"] == id]
        exploded_evaluations = pd.concat(
            [
                pd.DataFrame.from_dict(config_evaluations[col].values[0], orient="index", columns=["score"]).rename(
                    columns={"score": col}
                )
                for col in config_evaluations.columns
                if col != "id"
            ],
            axis=1,
        )
        print(exploded_evaluations)
        st.write(f"## `columnwise` metrics")
        st.dataframe(exploded_evaluations.style.highlight_max(axis=0))
        st.download_button(
            "Press to Download", convert_df(exploded_evaluations), "evaluations.csv", "text/csv", key="download-csv"
        )
        st.write("### Configurations")
        st.dataframe(
            experiments[metadata].drop(["seed", "repeat"], axis=1).groupby(["architecture", "config_idx"]).first()
        )


def pairwise_metrics(evaluations, experiments):
    st.write("## `pairwise` metrics")
    evaluation_df = pd.DataFrame(evaluations)
    metrics = [c for c in evaluation_df.columns if c != "id"]
    metrics_to_show = st.sidebar.multiselect("Select metric(s) to display", metrics, default=metrics)
    id = id_selector(experiments)
    for metric in metrics_to_show:
        st.write(f"## `{metric}`")
        grid = evaluation_df[evaluation_df["id"] == id][metric].values[0]
        st.table(
            pd.DataFrame(
                [(row, col, value["score"]) for (row, col), value in grid.items()], columns=["row", "col", "value"]
            )
            .set_index("row")
            .pivot(columns="col", values="value")
            .fillna(" ")
        )


def page():
    st.set_page_config(layout="wide")
    metric_groups = st.session_state["evaluations"].keys()
    selected_metric_group = st.sidebar.selectbox(
        "Select a metric group", [mg.capitalize() for mg in list(metric_groups)]
    ).lower()

    evaluations = st.session_state["evaluations"][selected_metric_group]
    experiments = st.session_state["experiments"]
    if selected_metric_group in ["table", "task", "privacy", "efficacy", "aequitas"]:
        table_metrics(evaluations, experiments, selected_metric_group)
    elif selected_metric_group == "columnwise":
        columnwise_metrics(evaluations, experiments)
    elif selected_metric_group == "pairwise":
        pairwise_metrics(evaluations, experiments)
    else:
        st.write("## Not yet implemented.")


if __name__ == "__main__":
    if "evaluations" not in st.session_state:
        st.error("Upload an evaluation bundle to get started!")
    else:
        page()
