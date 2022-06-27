import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda

class MultivariateNormalDiagWrapper(DistributionWrapper):
  def __init__(self, mean, var):
    super().__init__()
    std = tf.sqrt(var) + 1e-7
    # batch_shape: mean.shape[0] (n_rows)
    # event_shape: mean.shape[1] (n_cols)
    self._dist = tfp.distributions.MultivariateNormalDiag(
        mean, std, allow_nan_stats=False)
    self._parameters = dict()
    self._parameters.setdefault('mean', self._dist.mean())
    self._parameters.setdefault('var', self._dist.variance())

  @staticmethod
  def export_parameterizer(n_dims, name):
    decoders = dict()
    decoders.setdefault('mean', Sequential([Dense(n_dims)], name=f'{name}:mean'))
    decoders.setdefault('var', Sequential([
        Dense(n_dims), 
        Lambda(lambda x: tf.math.softplus(x))
    ], name=f'{name}:var'))
    
    return decoders