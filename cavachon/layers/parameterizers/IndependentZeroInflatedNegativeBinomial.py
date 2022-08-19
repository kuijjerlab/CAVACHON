from cavachon.distributions.Distribution import Distribution
import tensorflow as tf

class IndependentZeroInflatedNegativeBinomial(Distribution, tf.keras.layers.Layer):
  def __init__(
        self,
        event_dim: int,
        name: str = 'IndependentZeroInflatedNegativeBinomialParameterizer'):
    super().__init__(name=name)
    self.event_dim: int = event_dim
    return

  def build(self, input_shape):
    self.probs_weight = self.add_weight(
        f'{self.name}/probs_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.probs_bias = self.add_weight(
        f'{self.name}/probs_bias',
        shape=(1, self.event_dim))
    self.mean_weight = self.add_weight(
        f'{self.name}/mean_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.mean_bias = self.add_weight(
        f'{self.name}/mean_bias',
        shape=(1, self.event_dim))
    self.dispersion_weight = self.add_weight(
        f'{self.name}/std_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.dispersion_bias = self.add_weight(
        f'{self.name}/std_bias',
        shape=(1, self.event_dim))
    return

  def call(self, inputs) -> tf.Tensor:
    result = (
        tf.math.softmax(tf.matmul(inputs, self.probs_weight) + self.probs_bias),
        tf.matmul(inputs, self.mean_weight) + self.mean_bias,
        tf.math.softmax(tf.matmul(inputs, self.dispersion_weight) + self.dispersion_bias)
    )
    return tf.concat(result, axis=-1)