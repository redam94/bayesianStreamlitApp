import streamlit as st 

def get_var(var_name):
  return st.session_state.get(var_name, None)