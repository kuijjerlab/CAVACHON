from cavachon.distributions.MultivariateNormalDiagWrapper import MultivariateNormalDiagWrapper
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.losses.StandardKLDivergence import StandardKLDivergence

class NegativeElbo(tf.keras.losses.Loss):
  def __init__(self, module, modality_ordered_map, name='negative_evidence_lower_bound', **kwargs):
    kwargs.setdefault('name', name)
    super().__init__(**kwargs)
    self.module = module
    self.modality_ordered_map = modality_ordered_map
    self.negative_log_data_likelihood: NegativeLogDataLikelihood = NegativeLogDataLikelihood(
        self.module,
        self.modality_ordered_map)
    self.standard_kl_divergence: StandardKLDivergence = StandardKLDivergence(
        self.module,
        self.modality_ordered_map)
    self.cache: Dict[tf.Tensor] = tf.zeros((1,))

  def call(self, y_true, y_pred, sample_weight=None):
    negative_elbo = None
    negative_log_data_likelihood = self.negative_log_data_likelihood.call(
        y_true,
        y_pred, 
        sample_weight)
    standard_kl_divergence = self.standard_kl_divergence.call(
        y_true,
        y_pred,
        sample_weight)
    if negative_elbo is None:
      negative_elbo = negative_log_data_likelihood + standard_kl_divergence
    else:
      negative_elbo += negative_log_data_likelihood + standard_kl_divergence

    self.cache = tf.reduce_mean(negative_elbo)
    return negative_elbo

  def update_module(self, module):
    self.module = module
    self.negative_log_data_likelihood.update_module(module)
    self.standard_kl_divergence.update_module(module)
    return