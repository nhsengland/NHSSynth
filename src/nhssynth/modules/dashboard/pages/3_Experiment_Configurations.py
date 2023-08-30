import streamlit as st

if "experiments" not in st.session_state:
    st.error("Upload an evaluation bundle to get started!")
else:
    st.set_page_config(layout="wide")
    st.dataframe(st.session_state["experiments"])
