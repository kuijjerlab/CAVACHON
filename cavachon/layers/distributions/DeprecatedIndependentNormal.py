from cavachon.layers.distributions.DistributionLayer import DistributionLayer
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.layers.distributions.IndependentNormalWrapper import IndependentNormalWrapper

class IndependentNormal(DistributionLayer, tf.keras.layers.Layer):
  #DEPRECATED
  def __init__(
        self,
        name: str,
        n_dims: int):
    super().__init__(name=name)
    self.n_dims: int = n_dims
    return

  def build(self, input_shape):
    self.mean_parameterizer = self.add_weight(
        f'{self.name}/mean',
        shape=(int(input_shape[-1]), self.n_dims))
    self.pi_logit_parameterizer = self.add_weight(
        f'{self.name}/std',
        shape=(int(input_shape[-1]), self.n_dims))
    return

  def call(self, inputs) -> Dict[str, tf.Tensor]:
    result = dict()
    result.setdefault('mean', self.mean_parameterizer(inputs))
    result.setdefault('std', tf.math.softplus(self.pi_logit_parameterizer(inputs)))
    return result

  @staticmethod
  def dist(mean, std) -> tfp.distributions.Distribution:
    return tfp.distributions.Normal(mean, std)

  @staticmethod
  def prob(mean, std, value, name='prob', **kwargs) -> tf.Tensor:
    dist = IndependentNormal.dist(mean, std)
    return dist.prob(value, name, **kwargs)

  @staticmethod
  def sample(mean, std, training=False, seed=None, name='sample', **kwargs) -> tf.Tensor:
    dist = IndependentNormal.dist(mean, std)
    if training:
      return dist.sample((), seed, name, **kwargs)
    else:
      return dist.mean(name=name)