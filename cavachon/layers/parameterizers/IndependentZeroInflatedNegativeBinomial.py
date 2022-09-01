from cavachon.distributions.Distribution import Distribution
import tensorflow as tf

class IndependentZeroInflatedNegativeBinomial(tf.keras.layers.Layer):
  def __init__(
      self,
      event_dims: int,
      name: str = 'IndependentZeroInflatedNegativeBinomialParameterizer'):
    super().__init__(name=name)
    self.event_dims: int = event_dims
    return

  def build(self, input_shape: tf.TensorShape) -> None:
    self.logits_weight = self.add_weight(
        f'{self.name}/logits_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.logits_bias = self.add_weight(
        f'{self.name}/logits_bias',
        shape=(1, self.event_dims))
    self.mean_weight = self.add_weight(
        f'{self.name}/mean_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.mean_bias = self.add_weight(
        f'{self.name}/mean_bias',
        shape=(1, self.event_dims))
    self.dispersion_weight = self.add_weight(
        f'{self.name}/std_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.dispersion_bias = self.add_weight(
        f'{self.name}/std_bias',
        shape=(1, self.event_dims))
    return

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    dispersion = tf.math.sigmoid(tf.matmul(inputs, self.dispersion_weight) + self.dispersion_bias) + 1e-7
    dispersion = tf.where(dispersion == 0, 1e-7 * tf.ones_like(dispersion), dispersion)
    result = (
        tf.matmul(inputs, self.logits_weight) + self.logits_bias,
        tf.math.softmax(tf.matmul(inputs, self.mean_weight) + self.mean_bias),
        dispersion,
    )
    # shape: (batch, event_dims * 3)
    return tf.concat(result, axis=-1)