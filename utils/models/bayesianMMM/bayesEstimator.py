from abc import ABC

import pymc3 as pm 
import theano.tensor as tt
import theano
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.base as base
import sklearn.pipeline as pipeline


class Sampler:
  
  
  _ADVI_STEPS = 100_000
  _SAMPLER_MAP = {
    "Metropolis": pm.Metropolis, 
    "NUTS": pm.NUTS, 
    "HMC": pm.HamiltonianMC,
    "ADVI": pm.ADVI,
    "FullRankADVI": pm.FullRankADVI,
    }
  
  def __init__(self, sampler: str, sampler_kwargs: dict):
    if sampler not in self._SAMPLER_MAP.keys():
      raise ValueError("Unknown sampler")
    self.sampler = sampler
    self.sampler_kwargs = sampler_kwargs
    
  def sample(self, model, n_samples):
    with model:
      sample_method = self._SAMPLER_MAP[self.sampler]
      
      if self.sampler in ["ADVI", "FullRankADVI"]:
        self._inference = sample_method(**self.sampler_kwargs)
        self._approx_dist = pm.fit(self._ADVI_STEPS, method=self._inference)
        self.trace = self._approx_dist.sample(draws=n_samples)
        return self.trace
      self.trace = pm.sample(n_samples, tune=n_samples//2, step=sample_method(**self.sampler_kwargs))
      return self.trace
    
  def __repr__(self):
    return f"Sampler({self.sampler}, {self.sampler_kwargs})"
  
      
      
      
    
    
class BayesianLinearInterface(ABC, base.BaseEstimator):
  """Interface for linear bayesian models"""
  
  _trace: pm.backends.base.MultiTrace = None
  _model: pm.Model = None
  _coords: dict = None
  _sampler: Sampler = None
  _n_steps: int = 1_000
  
  def _define_model(self):
    pass
  
  def fit(self, X, y=None):
    if self._model is None:
      self._model = self._define_model()
    with self._model as model:
      if y is not None:
        pm.set_data({'input': X, 'observed': y})
      else:
        pm.set_data({'input': X})
      self._trace = self._sampler(model, self._n_steps)
    
    return self
        
      