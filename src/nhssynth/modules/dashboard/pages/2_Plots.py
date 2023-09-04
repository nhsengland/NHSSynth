import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import umap
from nhssynth.modules.dashboard.utils import (
    hide_streamlit_content,
    id_selector,
    subset_selector,
)
from sklearn.manifold import TSNE


def triangle(matrix):
    return matrix.where(np.tril(np.ones(matrix.shape), k=-1).astype(bool))


def distribution_plots(real_dataset, synthetic_datasets) -> None:
    column = st.sidebar.selectbox("Select column to display", real_dataset.columns)
    st.write(f"## Distribution Plot for `{column}`")
    synthetic_datasets = subset_selector(synthetic_datasets)
    fig = px.histogram(
        pd.concat(
            [
                pd.DataFrame({"Data": real_dataset[column], "Type": len(real_dataset) * ["Real"]}),
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "Data": sd.iloc[0][column],
                                "Type": len(sd.iloc[0]) * [f"Synthetic: {idx[0]} (Repeat {idx[1]}, Config {idx[2]})"],
                            }
                        )
                        for idx, sd in synthetic_datasets.iterrows()
                    ]
                ),
            ]
        ),
        x="Data",
        color="Type",
        barmode="overlay",
    )
    log_scale = st.sidebar.checkbox("Log scale", value=False)
    if log_scale:
        fig.update_layout(yaxis_type="log")
    if real_dataset[column].dtype == "object":
        sort_xaxis = st.sidebar.checkbox("Sort x-axis", value=True)
        if sort_xaxis:
            fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)


def correlation_plots(real_dataset, synthetic_datasets) -> None:
    show_continuous = st.sidebar.checkbox("Only show continuous columns", value=True)
    selection = real_dataset.select_dtypes(exclude=["object"]).columns
    if not show_continuous:
        non_continuous_columns = [c for c in real_dataset.columns if c not in selection]
        selection = st.sidebar.selectbox("Select categorical column to display", non_continuous_columns)
        if not non_continuous_columns:
            st.error("No categorical columns found!")
            return go.Figure()

    synthetic_dataset = id_selector(synthetic_datasets).iloc[0][selection]
    real_dataset = real_dataset[selection]
    if not show_continuous:
        synthetic_dataset = pd.get_dummies(synthetic_dataset)
        real_dataset = pd.get_dummies(real_dataset)

    correlation_type = st.sidebar.selectbox("Select correlation type", ["Pearson", "Spearman", "Kendall"]).lower()
    correlation_to_show = st.sidebar.selectbox("Select correlation to display", ["Real", "Synthetic", "Difference"])

    real_corr_matrix = real_dataset.corr(method=correlation_type)
    synthetic_corr_matrix = synthetic_dataset.corr(method=correlation_type)
    zmin = min(real_corr_matrix.min().min(), synthetic_corr_matrix.min().min())
    zmax = max(real_corr_matrix.max().max(), synthetic_corr_matrix.max().max())
    if correlation_to_show == "Difference":
        corr_matrix = triangle(abs(synthetic_corr_matrix - real_corr_matrix))
        zmin = 0
        zmax = zmax - zmin
    elif correlation_to_show == "Real":
        corr_matrix = triangle(real_corr_matrix)
    else:
        corr_matrix = triangle(synthetic_corr_matrix)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=px.colors.diverging.RdYlGn,
            reversescale=True,
            zmin=zmin,
            zmax=zmax,
        ),
        layout=go.Layout(xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def prepare_for_dimensionality(df: pd.DataFrame) -> pd.DataFrame:
    """Factorize all categorical columns in a dataframe."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]
        elif df[col].dtype == "datetime64[ns]":
            df[col] = pd.to_numeric(df[col])
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def plot_reducer(
    real_dataset: pd.DataFrame, synthetic_dataset: pd.DataFrame, reducer: Any, reducer_name: str
) -> go.Figure:
    with st.spinner(f"Running {reducer_name}..."):
        fig = go.Figure(layout=go.Layout(xaxis_title="UMAP 1", yaxis_title="UMAP 2"))
        proj_real = reducer.fit_transform(real_dataset)
        fig.add_scatter(
            x=proj_real[:, 0], y=proj_real[:, 1], mode="markers", marker=dict(size=5), opacity=0.75, name="Real data"
        )
        proj_synth = reducer.fit_transform(synthetic_dataset)
        fig.add_scatter(
            x=proj_synth[:, 0],
            y=proj_synth[:, 1],
            mode="markers",
            marker=dict(size=5),
            opacity=0.75,
            name="Synthetic data",
        )
    return fig


def dimensionality_plots(real_dataset: pd.DataFrame, synthetic_datasets: pd.DataFrame) -> None:
    synthetic_dataset = prepare_for_dimensionality(id_selector(synthetic_datasets).iloc[0].copy())
    real_dataset = prepare_for_dimensionality(real_dataset.copy())
    dimensionality_method = st.sidebar.selectbox("Select dimensionality reduction method", ["UMAP", "t-SNE"])
    run = st.sidebar.button("Run dimensionality reduction")
    if run:
        if dimensionality_method == "UMAP":
            reducer = umap.UMAP()
        if dimensionality_method == "t-SNE":
            reducer = TSNE(n_components=2, init="pca")
        fig = plot_reducer(real_dataset, synthetic_dataset, reducer, dimensionality_method)
        st.plotly_chart(fig, use_container_width=True)


def page():
    # metric_groups = st.session_state["evaluations"].keys()
    plot_types = ["Distribution", "Correlation", "Dimensionality"]
    selected_plot_type = st.sidebar.selectbox("Select a plot type", plot_types)

    real_data = st.session_state["typed"]
    synthetic_datasets = st.session_state["synthetic_datasets"]

    if selected_plot_type == "Distribution":
        distribution_plots(real_data, synthetic_datasets)
    elif selected_plot_type == "Correlation":
        correlation_plots(real_data, synthetic_datasets)
    elif selected_plot_type == "Dimensionality":
        dimensionality_plots(real_data, synthetic_datasets)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    hide_streamlit_content()
    if "evaluations" not in st.session_state:
        st.error("Upload an evaluation bundle to get started!")
    else:
        page()
