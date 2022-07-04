import pymc3 as pm
import numpy as np
import streamlit as st 


def create_model(df, media_col_names, dependent_col_names, control_col_names=None, geo_col_names=None, model=None):
  
  n_idx = len(df)
  
  n_media_vars = len(media_col_names)
  media_vars = df[media_col_names].values
  
  dependent = df[dependent_col_names].values
  
  n_geos = 1
  if geo_col_names:
    geos = df[geo_col_names].values.reshape(-1)
    n_geos = np.unique(geos).shape[0]
  
  n_control_vars = 1
  if control_col_names:
    control_vars = df[control_col_names].values
    n_control_vars = len(control_col_names)
    
  maxs = media_vars.max(axis=0)
  
  coords = {
    "geos": np.arange(n_geos),
    "ids": np.arange(n_idx),
    "media_cols": np.arange(n_media_vars),
    "control_cols": np.arange(n_control_vars)
  }
  
  model = pm.Model(coords=coords)
  
  with model:
    if geo_col_names:
      geo_idx = pm.Data("geographies", geos, dims="ids")
    
    media_vars = pm.Data("media_vars", media_vars , dims=["ids", "media_cols"])
    
    sigma = pm.HalfCauchy("sigma", 5)

    media_coeffs_mu = pm.HalfNormal("media_coeffs_mu", sigma=10, dims="media_cols")
    if geo_col_names:
      media_coeffs_geos = pm.Normal("media_coeffs_geos", sigma=1, dims=["geos", "media_cols"])
      media_coeffs_sigma = pm.HalfCauchy("media_coeffs_sigma", 5)
      media_coeffs = pm.Deterministic("media_coeffs", media_coeffs_mu + media_coeffs_geos*media_coeffs_sigma, dims=["geos", "media_cols"])
    else:
      media_coeffs = pm.Deterministic("media_coeffs", media_coeffs_mu, dims="media_cols")
    
    alphas = pm.Uniform("α", .8, 1, dims="media_cols")
    betas_coeff = pm.Uniform("β_coeff", 1, 15, dims="media_cols")
    betas =pm.Deterministic("β", 10**(-betas_coeff))

    
    media_transformed = betas**(alphas**(media_vars/maxs*100))
    if geo_col_names:
      media_contributions = pm.Deterministic("media_contributions", media_transformed*media_coeffs[geo_idx], dims=["ids", "media_cols"])
    else:
      media_contributions = pm.Deterministic("media_contributions", media_transformed*media_coeffs, dims=["ids", "media_cols"])
    incremental_sales_media = pm.math.sum(media_contributions, axis=1)
    
    total_sales = incremental_sales_media
    
    if control_col_names:
      control_vars = pm.Data("control_vars", control_vars, dims=["ids", "control_cols"])
      control_coeffs_mu = pm.HalfNormal("control_coeffs_mu", sigma=10, dims="control_cols")
      if geo_col_names:
        control_coeffs_geos = pm.Normal("control_coeffs_geos", sigma=1, dims=["geos", "control_cols"])
        control_coeffs_sigma = pm.HalfCauchy("control_coeffs_sigma", 5)
        control_coeffs = pm.Deterministic("control_coeffs", control_coeffs_mu + control_coeffs_geos*control_coeffs_sigma, dims=["geos", "control_cols"])
        control_contributions = pm.Deterministic("control_contributions", control_vars*control_coeffs[geo_idx], dims=["ids", "control_cols"])
      else:
        control_coeffs = pm.Deterministic("control_coeffs", control_coeffs_mu, dims="control_cols")
        control_contributions = pm.Deterministic("control_contributions", control_vars*control_coeffs, dims=["ids", "control_cols"])
      
      incremental_sales_control = pm.math.sum(control_contributions, axis=1)
      total_sales = pm.math.sum(incremental_sales_media + incremental_sales_control, axis=1)
      
    
    y = pm.Normal("incremental_sales_like", mu=total_sales, sigma=sigma, observed=dependent, dims="ids")
  return model


def run_inference(model):
  with model:
    inference = pm.FullRankADVI()
    approx = pm.fit(100_000, method=inference)
  return inference, approx