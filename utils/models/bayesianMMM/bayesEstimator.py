from abc import ABC

import pymc3 as pm 
import theano.tensor as tt
import theano
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.base as base
import sklearn.pipeline as pipeline


class Sampler:
  
  def make_adaptive_sampler(sampler):
    return lambda model, n_samples: sampler(model, n_samples, tune=1000, target_accept=0.9)
    
  _SAMPLER_MAP = {
    "Metropolis": pm.Metropolis, 
    "NUTS": pm.NUTS, 
    "HMC": pm.HamiltonianMC,
    "NUTS_adapt": make_adaptive_sampler(pm.NUTS), 
    "Metropolis_adapt": make_adaptive_sampler(pm.Metropolis),
    "HMC_adapt": make_adaptive_sampler(pm.HamiltonianMC),
    "ADVI": pm.ADVI,
    "FullRankADVI": pm.FullRankADVI,
    }
  
  def __init__(self, sampler: str):
    if sampler not in self._SAMPLER_MAP.keys():
      raise ValueError("Unknown sampler")
    self.sampler = sampler
    
  def sample(self, model, n_samples):
    with model:
      sample_method = self._SAMPLER_MAP[self.sampler]
      if self.sampler in ["ADVI", "FullRankADVI"]:
        inference = sample_method()
        approx_dist = inference.fit(n_samples, method=inference)
        return approx_dist
      
    
    
class BayesianLinearInterface(ABC, base.BaseEstimator):
  """Interface for linear bayesian models"""
  
  _trace: pm.backends.base.MultiTrace = None
  _model: pm.Model = None
  _coords: dict = None
  _sampler: Sampler = None
  
  def _define_model(self):
    pass
  
  def fit(self, X, y=None):
    if self._model is None:
      self._model = self._define_model()
    with self._model:
      if y is not None:
        pm.set_data({'input': X, 'observed': y})
      else:
        pm.set_data({'input': X})
      