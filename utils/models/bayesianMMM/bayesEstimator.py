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
  
  def __init__(self, sampler: str):
    if sampler not in self._SAMPLER_MAP.keys():
      raise ValueError("Unknown sampler")
    self.sampler = sampler
    
  def sample(self, model, n_samples):
    with model:
      sample_method = self._SAMPLER_MAP[self.sampler]
      
      if self.sampler in ["ADVI", "FullRankADVI"]:
        self._inference = sample_method()
        self._approx_dist = pm.fit(self._ADVI_STEPS, method=self._inference)
        self.trace = self._approx_dist.sample(draws=n_samples)
        return self.trace
      self.trace = pm.sample(n_samples, tune=n_samples//2, step=sample_method())
      
      
      
    
    
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
      