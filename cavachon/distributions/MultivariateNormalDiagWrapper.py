import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class MultivariateNormalDiagWrapper(DistributionWrapper):
  def __init__(self, mean, logvar):
    super().__init__()
    std = tf.sqrt(tf.exp(logvar)) + 1e-7
    # batch_shape: mean.shape[0] (n_rows)
    # event_shape: mean.shape[1] (n_cols)
    self._dist = tfp.distributions.MultivariateNormalDiag(
        mean, std, allow_nan_stats=False)
    self._parameters = dict()
    self._parameters.setdefault('mean', self._dist.mean())
    self._parameters.setdefault('variance', self._dist.variance())

  @staticmethod
  def export_parameterizer(n_dims, name):
    decoders = dict()
    decoders.setdefault('mean', Sequential([Dense(n_dims)], name=f'{name}:mean'))
    decoders.setdefault('logvar', Sequential([Dense(n_dims)], name=f'{name}:logvar'))
    
    return decoders