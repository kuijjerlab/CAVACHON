
import tensorflow as tf

from typing import Dict
from abc import ABC, abstractmethod, abstractstaticmethod

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

  @abstractstaticmethod
  def export_decoders(self) -> Dict[str, tf.keras.Sequential]:
    pass