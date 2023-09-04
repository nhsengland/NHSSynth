import streamlit as st
from nhssynth.modules.dashboard.utils import hide_streamlit_content

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    hide_streamlit_content()
    if "experiments" not in st.session_state:
        st.error("Upload an evaluation bundle to get started!")
    else:
        st.dataframe(st.session_state["experiments"])
