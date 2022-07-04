import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from cavachon.model.Parameterizer import Parameterizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Softmax

class IndependentZeroInflatedNegativeBinomialWrapper(DistributionWrapper):
  def __init__(self, pi_logit, mean, dispersion):
    super().__init__()
    # batch_shape: mean.shape
    # event_shape: ()
    # pi.shape: (mean.shape[0], mean.shape[1], 2)
    self._dist = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(logits=pi_logit),
        components=[
            tfp.distributions.Deterministic(tf.zeros_like(mean), allow_nan_stats=False),
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(mean, dispersion),
        ])

    self._parameters = dict()
    self._parameters.setdefault('pi', tf.keras.activations.softmax(pi_logit)[...,0])
    self._parameters.setdefault('mean', mean)
    self._parameters.setdefault('dispersion', dispersion)

  @staticmethod
  def export_parameterizer(n_dims, name):
    decoders = Parameterizer()
    decoders.setdefault('mean', Sequential([Dense(n_dims), Softmax()], name=f'{name}:mean'))
    decoders.setdefault('pi_logit', Sequential([
        Dense(n_dims * 2),
        Lambda(lambda x: tf.reshape(x, (-1, n_dims, 2)))
    ], name=f'{name}:pi_logit'))
    decoders.setdefault('dispersion', Sequential([Dense(n_dims), Softmax()], name=f'{name}:dispersion'))

    return decoders