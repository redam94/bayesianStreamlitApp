import streamlit as st
import streamlit.components as components
import utils

df = st.session_state.get("df", None)

geos = st.session_state.get("geos_old", None)
media_vars = st.session_state.get("media_vars_old", None)
control_vars = st.session_state.get("control_vars_old", None)
dependent_var = st.session_state.get("dependent_var_old", None)
period_var = st.session_state.get("period_var_old", None)
index = 0
if df is not None:
  for i in range(len(df.columns)):
    index = i
    if df.columns[i] == dependent_var:
      break
  st.session_state["index"] = index

def clear_trace_inference():
  st.session_state["inference"] =None
  st.session_state["trace"] = None
  st.session_state["fig"] = None
if df is not None:
  column_names = df.columns.tolist()
  with st.form("selection-form"):
    
    media_vars_new = st.multiselect("Select Media Variables", column_names, key="media_vars", default=media_vars)
    control_vars_new = st.multiselect("Select Control Variables", column_names, key="control_vars", default=control_vars)
    dependent_var_new = st.selectbox("Select Dependent Variable", column_names, key="dependent_var", index=index)
    period_var_new = st.selectbox("Select Period Variable", column_names, key="period_var")
    geos_new = st.multiselect("Select Geography Columns", column_names, key="geos", default=geos)
    if geos_new is not None:
      num_geos=1
      for geo in geos_new:
        num_geos *= df[geo].nunique()
      st.markdown(f"Number of Unique Geographies is {num_geos}")
    
    st.form_submit_button("Submit", on_click=clear_trace_inference)
    
  if (media_vars_new or control_vars_new) and dependent_var_new and period_var_new:
    st.session_state["media_vars_old"] = media_vars_new
    st.session_state["control_vars_old"] = control_vars_new
    st.session_state["dependent_var_old"] = dependent_var_new
    st.session_state["period_var_old"] = period_var_new
    st.session_state["geos_old"] = geos_new
    st.markdown(f"""
                ### Current Selections:
                {"Media Variables: " if media_vars else "" } {media_vars_new}
                """)
    with st.spinner("Creating Model"):
      st.session_state["model"] = utils.models.bayesianMMM.create_model(df, media_vars_new, dependent_var_new, control_vars_new, geos_new)