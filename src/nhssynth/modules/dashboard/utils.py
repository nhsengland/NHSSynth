import pandas as pd
import streamlit as st


def hide_streamlit_content() -> None:
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def id_selector(df: pd.DataFrame) -> pd.Series:
    architecture = st.sidebar.selectbox(
        "Select architecture to display", df.index.get_level_values("architecture").unique()
    )
    repeats = df.loc[architecture].index.get_level_values("repeat").astype(int).unique()
    configs = df.loc[architecture].index.get_level_values("config").astype(int).unique()
    if len(repeats) > 1:
        repeat = st.sidebar.selectbox("Select repeat to display", repeats)
    else:
        repeat = repeats[0]
    if len(configs) > 1:
        config = st.sidebar.selectbox("Select configuration to display", configs)
    else:
        config = configs[0]
    return df.loc[(architecture, repeat, config)]


def subset_selector(df: pd.DataFrame) -> pd.DataFrame:
    architectures = df.index.get_level_values("architecture").unique().tolist()
    repeats = df.index.get_level_values("repeat").astype(int).unique().tolist()
    configs = df.index.get_level_values("config").astype(int).unique().tolist()
    selected_architectures = st.sidebar.multiselect(
        "Select architectures to display", architectures, default=architectures
    )
    selected_repeats = st.sidebar.multiselect("Select repeats to display", repeats, default=repeats[0])
    selected_configs = st.sidebar.multiselect("Select configurations to display", configs, default=configs)
    return df.loc[(selected_architectures, selected_repeats, selected_configs)]
