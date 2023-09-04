import argparse
import os
import pickle
from typing import Any

import streamlit as st
from nhssynth.modules.dashboard.utils import hide_streamlit_content
from nhssynth.modules.dataloader.io import TypedDataset
from nhssynth.modules.evaluation.io import Evaluations
from nhssynth.modules.model.io import Experiments, SyntheticDatasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHSSynth Evaluation Dashboard")
    parser.add_argument("--evaluations", type=str, help="Path to a set of evaluations.")
    parser.add_argument("--experiments", type=str, help="Path to a set of experiments.")
    parser.add_argument("--synthetic-datasets", type=str, help="Path to a set of synthetic datasets.")
    parser.add_argument("--typed", type=str, help="Path to a typed real dataset.")
    return parser.parse_args()


def get_component(name: str, component_type: Any, text: str):
    uploaded = st.file_uploader(f"Upload a pickle file containing a {text}", type="pkl")
    if getattr(args, name):
        with open(os.getcwd() + "/" + getattr(args, name), "rb") as f:
            loaded = pickle.load(f)
    if uploaded is not None:
        loaded = pickle.load(uploaded)
    if loaded is not None:
        assert isinstance(loaded, component_type), f"Uploaded file does not contain a {text}!"
        st.session_state[name] = loaded.contents
        st.success(f"Loaded {text}!")


if __name__ == "__main__":
    args = parse_args()

    st.set_page_config(page_title="NHSSynth Evaluation Dashboard", page_icon="👋")
    hide_streamlit_content()
    st.title("NHSSynth Evaluation Dashboard")
    st.write(
        "Welcome! Upload an evaluation bundle below to get started (optionally also the typed real dataset and bundle of experiments containing the synthetic datasets).\n\nUse the menu on the left to navigate the dashboard."
    )

    get_component("evaluations", Evaluations, "bundle of evaluations")
    get_component("experiments", Experiments, "bundle of experiments")
    get_component("synthetic_datasets", SyntheticDatasets, "bundle of synthetic datasets")
    get_component("typed", TypedDataset, "typed real dataset")
