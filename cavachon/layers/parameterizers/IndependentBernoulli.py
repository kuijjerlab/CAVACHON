from typing import Tuple

import tensorflow as tf

class IndependentBernoulli(tf.keras.layers.Layer):
  def __init__(
      self,
      event_dims: Tuple[int],
      name: str = 'IndependentBernoulliParameterizer'):
    super().__init__(name=name)
    self.event_dims: Tuple[int] = event_dims
    return

  def build(self, input_shape: tf.TensorShape) -> None:
    self.logits_weight = self.add_weight(
        f'{self.name}/logits_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.logits_bias = self.add_weight(
        f'{self.name}/logits_bias',
        shape=(1, self.event_dims))
    return

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    # shape: (batch, event_dims)
    return tf.matmul(inputs, self.logits_weight) + self.logits_bias