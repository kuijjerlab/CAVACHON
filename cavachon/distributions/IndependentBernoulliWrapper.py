import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class IndependentBernoulliWrapper(DistributionWrapper):
  def __init__(self, pi_logit):
    super().__init__()
    # batch_shape: logits.shape
    # event_shape: ()
    self._dist = tfp.distributions.Bernoulli(logits=pi_logit)
    self._parameters = dict()
    self._parameters.setdefault('pi', tf.keras.activations.sigmoid(pi_logit))

  @staticmethod
  def export_parameterizer(n_dims, name):
    decoders = dict()
    decoders.setdefault('pi_logit', Sequential([Dense(n_dims)], name=f'{name}:pi_logit'))
    
    return decoders