import arviz as az 
import matplotlib.pyplot as plt

def plot_trace(model, trace, var_names=None):
  
  if var_names is None:
    var_names = ["α", "β_coeff", "media_coeffs"]
  
  n_plots = len(var_names)
  fig, ax = plt.subplots(n_plots, 2, figsize=(16, n_plots*8))
  
  with model:
    az.plot_trace(trace, var_names=var_names, axes=ax)
  
  return fig