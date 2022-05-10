import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.Distribution import Distribution
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class MultivariateNormalDiag(Distribution):
  def __init__(self, mean, logvar):
    super().__init__()
    std = tf.sqrt(logvar) + 1e-7
    self._dist = tfp.distributions.MultivariateNormalDiag(
        mean, std, allow_nan_stats=False)
    self._parameters = dict()
    self._parameters.setdefault('mean', self._dist.mean())
    self._parameters.setdefault('variance', self._dist.variance())

  @staticmethod
  def export_decoders(n_dims):
    decoders = dict()
    decoders.setdefault('mean', Sequential([Dense(n_dims)]))
    decoders.setdefault('logvar', Sequential([Dense(n_dims)]))
    
    return decoders