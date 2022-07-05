
from __future__ import annotations
from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.losses.NegativeElbo import NegativeElbo
from cavachon.metrics.CustomMetrics import CustomMetrics
from typing import Any

import tensorflow as tf
import warnings

class LogDataLikelihood(CustomMetrics, tf.keras.metrics.Metric):
  def __init__(self, name: str = 'log_data_likelihood', **kwargs):
    super().__init__(name=name, **kwargs)
    self.metric = 0

  def update_state(self, y_true: Any, y_pred: Any, modality_ordered_map: ModalityOrderedMap, **kwargs) -> None:
    x_parameters = y_pred.get('x_parameters')
    
    log_data_likelihood = None

    for modality_name, modality in modality_ordered_map.data.items():
      # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] 
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
      # term (a): logpx_z
      x = tf.sparse.to_dense(y_true.get(f'{modality_name}:matrix'))
      dist_x_z = modality.dist_cls(**x_parameters.get(modality_name))
      logpx_z = tf.reduce_sum(dist_x_z.log_prob(x), axis=-1)
      
      if log_data_likelihood is None:
        log_data_likelihood = logpx_z
      else:
        log_data_likelihood += logpx_z

    self.metric = tf.reduce_mean(log_data_likelihood)   
    return

  def update_state_from_loss_cache(self, custom_loss: CustomLoss) -> None:
    if isinstance(custom_loss, NegativeElbo):
      self.metric = -1 * custom_loss.negative_log_data_likelihood.cache
    elif isinstance(custom_loss, NegativeLogDataLikelihood):
      self.metric = -1 * custom_loss.cache
    else:
      message = "".join((
        f"{self.__class__.__name__} expect loss cache of either {NegativeElbo.__name__} ",
        f"or {NegativeLogDataLikelihood.__name__}. Get {custom_loss.__class__.__name__}."
      ))
      warnings.warn(message, RuntimeWarning)

    return