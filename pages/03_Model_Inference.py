import streamlit as st
import streamlit.components as components
import utils

df = st.session_state.get("df", None)
media_vars = st.session_state.get("media_vars", None)
control_vars = st.session_state.get("control_vars", None)
dependent_var = st.session_state.get("dependent_var", None)
period_var = st.session_state.get("period_var", None)
geos = st.session_state.get("geos", None)
model = st.session_state.get("model", None)
fig = None
  
if model is not None:
  with st.spinner("Running Inference"):
    inference, approx = utils.models.bayesianMMM.run_inference(model)
    st.session_state["inference"] = inference
    st.session_state["approx"] = approx
  with st.spinner("Sampling traces"):
    trace = approx.sample(draws=10_000)
    st.session_state["trace"] = trace
  with st.spinner("Plotting traces"):
    fig_html, fig = utils.visualizer.plot_trace(model, trace)
    #components.v1.html(fig_html, height=2000)
    
if fig is not None:
  with st.expander("Show trace plot"):
    st.pyplot(fig)
