
import tensorflow as tf

from abc import ABC, abstractmethod, abstractstaticmethod
from cavachon.model.Parameterizer import Parameterizer
from typing import Dict

class DistributionWrapper(ABC):

  @property
  def dist(self):
    return self._dist
 
  @dist.setter
  def dist(self, val):
    self._dist = val

  @property
  def parameters(self):
    return self._parameters

  @parameters.setter
  def parameters(self, val):
    self._parameters = val

  def log_prob(self, *args, **kwargs):
    return self._dist.log_prob(*args, **kwargs)

  def prob(self, *args, **kwargs):
    return self._dist.prob(*args, **kwargs)

  @abstractstaticmethod
  def export_parameterizer(n_dims, name) -> Parameterizer:
    pass

  