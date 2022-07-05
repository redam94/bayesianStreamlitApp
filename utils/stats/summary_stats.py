import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az

def summarize_trace(trace, var_names, model=None):
  if model is None:
    return az.summary(trace, var_names=var_names)
  with model:
    summary = az.summary(trace, var_names=var_names)
  return summary
  