from typing import Tuple

import tensorflow as tf

class IndependentBernoulli(tf.keras.layers.Layer):
  def __init__(
      self,
      event_dim: Tuple[int],
      name: str = 'IndependentBernoulliParameterizer'):
    super().__init__(name=name)
    self.event_dim: Tuple[int] = event_dim
    return

  def build(self, input_shape):
    self.probs_weight = self.add_weight(
        f'{self.name}/probs_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.probs_bias = self.add_weight(
        f'{self.name}/probs_bias',
        shape=(1, self.event_dim))
    return

  def call(self, inputs) -> tf.Tensor:
    return tf.math.softmax(tf.matmul(inputs, self.probs_weight) + self.probs_bias)