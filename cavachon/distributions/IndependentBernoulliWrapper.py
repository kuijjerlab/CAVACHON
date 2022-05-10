import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.DistributionWrapper import DistributionWrapper
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class IndependentBernoulliWrapper(DistributionWrapper):
  def __init__(self, logits):
    super().__init__()
    # batch_shape: logits.shape
    # event_shape: ()
    self._dist = tfp.distributions.Bernoulli(logits=logits)
    self._parameters = dict()
    self._parameters.setdefault('pi', tf.keras.activations.sigmoid(logits))

  @staticmethod
  def export_decoders(n_dims):
    decoders = dict()
    decoders.setdefault('pi', Sequential([Dense(n_dims)]))
    
    return decoders