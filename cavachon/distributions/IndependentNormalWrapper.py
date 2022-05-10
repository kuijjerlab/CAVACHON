import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Softmax

class IndependentNormalWrapper(DistributionWrapper):
  def __init__(self, mean, logvar):
    super().__init__()
    # batch_shape: mean.shape
    # event_shape: ()
    std = tf.sqrt(tf.exp(logvar)) + 1e-7
    self._dist = tfp.distributions.Normal(mean, std)

    self._parameters = dict()
    self._parameters.setdefault('mean', mean)
    self._parameters.setdefault('std', std)

  @staticmethod
  def export_decoders(n_dims):
    decoders = dict()
    decoders.setdefault('mean', Sequential([
        Dense(n_dims)
    ]))
    decoders.setdefault('logvar', Sequential([
        Dense(n_dims)
    ])),

    return decoders