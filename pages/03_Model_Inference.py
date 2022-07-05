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

fig = st.session_state.get("fig", None)
inference = st.session_state.get("inference", None)
trace = st.session_state.get("trace", None)

if model is not None:
  if inference is None:
    with st.spinner("Running Inference"):
      inference, approx = utils.models.bayesianMMM.run_inference(model)
      st.session_state["inference"] = inference
      st.session_state["approx"] = approx
  if trace is None:
    with st.spinner("Sampling traces"):
      trace = approx.sample(draws=10_000)
      st.session_state["trace"] = trace
      st.session_state["trace_summary"] = utils.stats.summarize_trace(trace, ["α", "β_coeff", "media_coeffs"], model)
  if fig is None: 
    with st.spinner("Plotting traces"):
      fig_html, fig = utils.visualizer.plot_trace(model, trace)
      st.session_state["fig"] = fig
    #components.v1.html(fig_html, height=2000)
    
if fig is not None:
  with st.expander("Show trace plot"):
    st.pyplot(fig)

if st.session_state.get("trace_summary", None) is not None:
  with st.expander("Show inference results"):
    st.write(st.session_state["trace_summary"])