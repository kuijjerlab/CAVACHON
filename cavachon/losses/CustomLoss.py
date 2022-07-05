from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

import tensorflow as tf

class CustomLoss(ABC):
  @property
  def cache(self):
    return self._cache
 
  @cache.setter
  def cache(self, val):
    self._cache = val 
  
  @abstractmethod
  def call(
      self,
      y_true: Any,
      y_pred: Any,
      sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    return 

  @abstractmethod
  def update_module(self, module: Module) -> None:
    return
