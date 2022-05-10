import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.Distribution import Distribution
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Softmax

class ZeroInflatedNegativeBinomial(Distribution):
  def __init__(self, pi, mean, dispersion):
    super().__init__()

    self._dist = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(logits=pi),
        components=[
            tfp.distributions.Deterministic(tf.zeros_like(mean), allow_nan_stats=False),
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(mean, dispersion),
        ])

    self._parameters = dict()
    self._parameters.setdefault('pi', pi)
    self._parameters.setdefault('mean', self._dist.mean())
    self._parameters.setdefault('dispersion', dispersion)

  @staticmethod
  def create_decoders(n_dims):
    decoders = dict()
    decoders.setdefault('mean', Sequential([
        Dense(n_dims)
    ]))
    decoders.setdefault('pi', Sequential([
        Dense(n_dims * 2),
        Lambda(lambda x: tf.reshape(x, (-1, n_dims, 2)))
    ]))
    decoders.setdefault('dispersion', Sequential([
        Softmax()
    ])),

    return decoders