
from __future__ import annotations
from cavachon.losses.NegativeElbo import NegativeElbo
from cavachon.metrics.CustomMetrics import CustomMetrics
from cavachon.metrics.LogDataLikelihood import LogDataLikelihood
from cavachon.metrics.KLDivergence import KLDivergence
from cavachon.model.Module import Module
from typing import Any

import tensorflow as tf
import warnings

class Elbo(CustomMetrics, tf.keras.metrics.Metric):
  def __init__(self, name: str = 'elbo', **kwargs):
    super().__init__(name=name, **kwargs)
    self.log_data_likelihood = LogDataLikelihood()
    self.standard_kl_divergence = KLDivergence()
    self.metric = 0

  def update_state(self, y_true: Any, y_pred: Any, module: Module, modality_ordered_map: ModalityOrderedMap, **kwargs) -> None:
    self.log_data_likelihood.update_state(y_true, y_pred, modality_ordered_map)
    self.standard_kl_divergence(y_pred, module, modality_ordered_map)
    self.metric = self.log_data_likelihood.result() - self.standard_kl_divergence.result()
    return

  def update_state_from_loss_cache(self, custom_loss: CustomLoss) -> None:
    if isinstance(custom_loss, NegativeElbo):
      self.log_data_likelihood.update_state_from_loss_cache(custom_loss)
      self.standard_kl_divergence.update_state_from_loss_cache(custom_loss)
      self.metric = -1 * custom_loss.cache
    else:
      message = "".join((
        f"{self.__class__.__name__} expect loss cache {NegativeElbo.__name__}. Get",
        f"{custom_loss.__class__.__name__}."
      ))
      warnings.warn(message, RuntimeWarning)

    return