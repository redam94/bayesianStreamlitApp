import arviz as az 
import matplotlib.pyplot as plt
import numpy as np
import mpld3

def plot_trace(model, trace, var_names=None):
  
  if not var_names:
    var_names = None
  n_plots=len(model.named_vars.keys())
  if var_names:
    n_plots = len(var_names)
  fig, ax = plt.subplots(n_plots, 2, figsize=(16, n_plots*8))
  if n_plots ==1:
    ax = ax[np.newaxis, :]
  with model:
    az.plot_trace(trace, var_names=var_names, axes=ax)
  fig_html = mpld3.fig_to_html(fig)
  return fig_html, fig

def save_html(fig_html):
  pass