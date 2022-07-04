import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import arviz as az
import streamlit as st
import streamlit.components as components
import utils

@st.cache
def read_data(file):
  df = pd.read_csv(file)
  return df

@st.cache
def convert_df(df):
  return df.to_csv().encode('utf-8')

def main(file):
  pass

st.markdown("""
  # Bayesian Linear Regression
  ---
  ## 1. Upload data
  """)

upload_file = None
geos = None
media_vars = None
control_vars = None
dependent_var = None
period_var = None
model = None

if upload_file is None:
  upload_file = st.file_uploader("Upload a CSV file", type=".csv")

if upload_file:
  st.markdown("## 2. Data Preview")
  df = read_data(upload_file)
  st.dataframe(df.head())
  st.markdown("## 3. Select Columns")
  
  with st.form("selection-form"):
    
    media_vars = st.multiselect("Select Media Variables", df.columns)
    control_vars = st.multiselect("Select Control Variables", df.columns)
    dependent_var = st.selectbox("Select Dependent Variable", df.columns)
    period_var = st.selectbox("Select Period Variable", df.columns)
    geos = st.multiselect("Select Geography Columns", df.columns)
    if geos:
      num_geos=1
      for geo in geos:
        num_geos *= df[geo].nunique()
      st.markdown(f"Number of Unique Geographies is {num_geos}")
      
    st.form_submit_button("Submit")
  
  if media_vars and dependent_var and period_var:
    with st.spinner("Creating Model"):
      model = utils.models.bayesianMMM.create_model(df, media_vars, dependent_var, control_vars, geos)
  
  if model:
    with st.spinner("Running Inference"):
      inference, approx = utils.models.bayesianMMM.run_inference(model)
    
    with st.spinner("Sampling traces"):
      trace = approx.sample(draws=10_000)
    
    with st.spinner("Plotting traces"):
      fig_html, fig = utils.visualizer.plot_trace(model, trace)
      #components.v1.html(fig_html, height=2000)
      st.pyplot(fig)
