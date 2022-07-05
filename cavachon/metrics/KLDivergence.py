from __future__ import annotations
from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper
from cavachon.losses.NegativeElbo import NegativeElbo
from cavachon.losses.KLDivergence import KLDivergence as KLDivergenceLoss
from cavachon.metrics.CustomMetrics import CustomMetrics
from cavachon.model.Module import Module
from typing import Any


import tensorflow as tf
import tensorflow_probability as tfp
import warnings

class KLDivergence(CustomMetrics, tf.keras.metrics.Metric):
  def __init__(self, name: str = 'kullback_leibler_divergence', **kwargs):
    super().__init__(name=name, **kwargs)
    self.metric = 0

  def update_state(self, y_pred: Any, module: Module, modality_ordered_map: ModalityOrderedMap, **kwargs) -> None:
    z_parameters = y_pred.get('z_parameters')
    z = y_pred.get('z')
    
    kl_divergence = None
    for modality_name, modality in modality_ordered_map.data.items():
      # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)] 
      # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)] 
      z_prior = module.z_prior.get(modality_name)
      dist_z_x = MultivariateNormalDiagWrapper(**z_parameters.get(modality_name))
      dist_z_y = MultivariateNormalDiagWrapper(
          tf.transpose(z_prior.mean_z_y, [1, 0]),
          tf.transpose(z_prior.var_z_y, [1, 0]))
      dist_y = tfp.distributions.Categorical(
          logits=tf.squeeze(z_prior.pi_logit_y),
          allow_nan_stats=False)
      dist_z = tfp.distributions.MixtureSameFamily(dist_y, dist_z_y.dist)
      
      logpz_y = dist_z_y.log_prob(tf.expand_dims(z.get(modality_name), -2))
      logpy = tf.math.log(z_prior.pi_y + 1e-7)
      logpz = dist_z.log_prob(tf.expand_dims(z.get(modality_name), -2))
      
      logpy_z = logpz_y + logpy - logpz
      py_z = tf.exp(logpy_z)
      
      # term (b): 𝚺_j𝚺_y[py_z(logpz_y)]
      py_z_logpz_y = tf.reduce_sum(py_z * logpz_y, axis=-1)

      # term (c): 𝚺_j𝚺_y[py_z(logpy)]
      py_z_logpy = tf.reduce_sum(py_z * logpy, axis=-1)

      # term (d): 𝚺_j[logqz_x]
      logqz_x = dist_z_x.log_prob(z.get(modality_name))
      
      # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
      py_z_logpy_z = tf.reduce_sum(py_z * logpy_z, axis=-1)
  
      kl_divergence = -py_z_logpz_y if kl_divergence is None else kl_divergence - py_z_logpz_y
      kl_divergence += -py_z_logpz_y
      kl_divergence += -py_z_logpy
      kl_divergence += py_z_logpy_z
      kl_divergence += logqz_x

    self.metric = tf.reduce_mean(kl_divergence)
    return

  def update_state_from_loss_cache(self, custom_loss: CustomLoss) -> None:
    if isinstance(custom_loss, NegativeElbo):
      self.metric = custom_loss.standard_kl_divergence.cache
    elif isinstance(custom_loss, KLDivergenceLoss):
      self.metric = custom_loss.cache
    else:
      message = "".join((
        f"{self.__class__.__name__} expect loss cache of either {NegativeElbo.__name__} ",
        f"or {KLDivergenceLoss.__name__}. Get {custom_loss.__class__.__name__}."
      ))
      warnings.warn(message, RuntimeWarning)

    return