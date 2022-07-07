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
var_names = st.session_state.get("var_names", None)
fig = st.session_state.get("fig", None)
inference = st.session_state.get("inference", None)
trace = st.session_state.get("trace", None)

def var_names_form_control():
  st.session_state["fig"] = None
  return

if model is not None:
  
  with st.form("var-names-form"):
    var_names = st.multiselect("Select Variables to explore", [x for x in list(model.named_vars.keys()) if not '__' in x], key="var_names")
    st.form_submit_button("Submit", on_click=var_names_form_control)
  
      
  if fig is None and var_names: 
    with st.spinner("Plotting traces"):
      st.session_state["trace_summary"] = utils.stats.summarize_trace(trace, var_names, model)
      fig_html, fig = utils.visualizer.plot_trace(model, trace, var_names)
      st.session_state["fig"] = fig
    #components.v1.html(fig_html, height=2000)
    
  if fig is not None:
    with st.expander("Show trace plot"):
      st.pyplot(fig)

  if st.session_state.get("trace_summary", None) is not None:
    with st.expander("Show inference results"):
      st.write(st.session_state["trace_summary"])