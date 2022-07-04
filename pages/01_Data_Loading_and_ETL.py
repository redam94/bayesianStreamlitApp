import pandas as pd
import pandas_profiling as pp
import streamlit as st
import streamlit_pandas_profiling as stprofile
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

@st.cache(allow_output_mutation=True)
def get_profile(df):
  profile = pp.ProfileReport(df, dark_mode=True, sensitive=True)
  return profile

if upload_file is None:
  upload_file = st.file_uploader("Upload a CSV file", type=".csv", key="upload_file")

if upload_file:
  st.markdown("## 2. Data Preview")
  df = read_data(st.session_state["upload_file"])
  st.session_state["df"] = df
  st.dataframe(df)
  profile = get_profile(df)
  with st.expander("Report", expanded=True):
    stprofile.st_profile_report(profile)

  

  
  