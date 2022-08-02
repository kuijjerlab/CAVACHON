from cavachon.layers.distributions.DistributionLayer import DistributionLayer
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

class IndependentBernoulli(DistributionLayer, tf.keras.layers.Layer):
  def __init__(
        self,
        name: str,
        n_dims: int):
    super().__init__(name=name)
    self.n_dims: int = n_dims
    return

  def build(self, input_shape):
    self.pi_logit_parameterizer = self.add_weight(
        f'{self.name}/pi_logit',
        shape=(int(input_shape[-1]), self.n_dims))
    return

  def call(self, inputs) -> Dict[str, tf.Tensor]:
    result = dict()
    result.setdefault(f'pi_logit', self.pi_logit_parameterizer(inputs))
    return result

  @staticmethod
  def dist(pi_logit) -> tfp.distributions.Distribuion:
    # batch_shape: pi_logit.shape
    # event_shape: ()
    return tfp.distributions.Bernoulli(logits=pi_logit)

  @staticmethod
  def prob(pi_logit, value, name='prob', **kwargs) -> tf.Tensor:
    dist = IndependentBernoulli.dist(pi_logit)
    return dist.prob(value, name, **kwargs)

  @staticmethod
  def sample(pi_logit, training=False, seed=None, name='sample', **kwargs) -> tf.Tensor:
    dist = IndependentBernoulli.dist(pi_logit)
    if training:
      return dist.sample((), seed, name, **kwargs)
    else:
      return tf.where(pi_logit > 0, tf.ones_like(pi_logit), tf.zeros_like(pi_logit))