from __future__ import annotations
from abc import ABC, abstractmethod

import tensorflow as tf

class CustomMetrics(ABC):

  @property
  def metric(self) -> tf.Tensor:
    return self._metric
 
  @metric.setter
  def metric(self, val) -> None:
    self._metric = val
  
  def result(self) -> tf.Tensor:
    return self._metric

  def reset_state(self) -> None:
    self._metric = 0
    return

  @abstractmethod
  def update_state(self, *args, **kwargs) -> None:
    return

  @abstractmethod
  def update_state_from_loss_cache(self, custom_loss: CustomLoss) -> None:
    return