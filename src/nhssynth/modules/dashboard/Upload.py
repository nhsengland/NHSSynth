import argparse
import os
import pickle

import pandas as pd
import streamlit as st
from nhssynth.modules.dataloader.metatransformer import TypedDataset
from nhssynth.modules.evaluation.utils import EvalBundle

parser = argparse.ArgumentParser(description="NHSSynth Evaluation Dashboard")
parser.add_argument("--evaluation-bundle", type=str, help="Path to an evaluation bundle pickle file.")
parser.add_argument("--experiments", type=str, help="Path to a set of experiments.")
parser.add_argument("--typed", type=str, help="Path to a typed real dataset.")
args = parser.parse_args()

st.set_page_config(page_title="NHSSynth Evaluation Dashboard", page_icon="👋")

st.title("NHSSynth Evaluation Dashboard")

st.write(
    "Welcome! Upload an evaluation bundle below to get started (optionally also the typed real dataset and bundle of experiments containing the synthetic datasets).\n\nUse the menu on the left to navigate the dashboard."
)

uploaded_eval_bundle = st.file_uploader("Upload a pickle file containing an evaluation bundle", type="pkl")
if args.evaluation_bundle:
    with open(os.getcwd() + "/" + args.evaluation_bundle, "rb") as f:
        eval_bundle = pickle.load(f)
if uploaded_eval_bundle is not None:
    eval_bundle = pickle.load(uploaded_eval_bundle)
if eval_bundle is not None:
    assert isinstance(eval_bundle, EvalBundle), "Uploaded file does not contain an evaluation bundle!"
    st.session_state["evaluations"], st.session_state["experiments"] = eval_bundle.evaluations, eval_bundle.experiments
    st.success(f"Loaded evaluation bundle!")

uploaded_experiments = st.file_uploader("Upload a pickle file containing a set of experiments", type="pkl")
if args.experiments:
    with open(os.getcwd() + "/" + args.experiments, "rb") as f:
        experiments = pickle.load(f)
if uploaded_experiments is not None:
    experiments = pickle.load(uploaded_experiments)
if experiments is not None:
    experiments = pd.DataFrame(experiments)
    assert (
        "dataset" in experiments.columns and "id" in experiments.columns
    ), "Uploaded file does not contain a set of experiments!"
    st.session_state["synthetic_data"] = experiments[["id", "dataset"]]
    st.success(f"Loaded synthetic datasets from experiments!")

uploaded_typed = st.file_uploader(
    "Upload a pickle file containing the typed (by the dataloader module) real dataset", type="pkl"
)
if args.typed:
    with open(os.getcwd() + "/" + args.typed, "rb") as f:
        typed = pickle.load(f)
if uploaded_typed is not None:
    typed = pickle.load(uploaded_typed)
if typed is not None:
    assert isinstance(typed, TypedDataset), "Uploaded file does not contain a typed real dataset!"
    st.session_state["real_data"] = typed
    st.success(f"Loaded real dataset!")
