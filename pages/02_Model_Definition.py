import streamlit as st
import streamlit.components as components
import utils

df = st.session_state.get("df", None)

if df is not None:
  with st.form("selection-form"):
    
    media_vars = st.multiselect("Select Media Variables", df.columns, key="media_vars")
    control_vars = st.multiselect("Select Control Variables", df.columns, key="control_vars")
    dependent_var = st.selectbox("Select Dependent Variable", df.columns, key="dependent_var")
    period_var = st.selectbox("Select Period Variable", df.columns, key="period_var")
    geos = st.multiselect("Select Geography Columns", df.columns, key="geos")
    if geos:
      num_geos=1
      for geo in geos:
        num_geos *= df[geo].nunique()
      st.markdown(f"Number of Unique Geographies is {num_geos}")
    
    st.form_submit_button("Submit")
    
  if media_vars and dependent_var and period_var:
    with st.spinner("Creating Model"):
      st.session_state["model"] = utils.models.bayesianMMM.create_model(df, media_vars, dependent_var, control_vars, geos)