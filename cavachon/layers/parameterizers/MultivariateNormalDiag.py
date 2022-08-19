import tensorflow as tf

class MultivariateNormalDiag(tf.keras.layers.Layer):
  def __init__(
      self,
      event_dim: int,
      name: str = 'MultivariateNormalDiagParameterizer'):
    super().__init__(name=name)
    self.event_dim: int = event_dim
    return

  def build(self, input_shape):
    self.loc_weight = self.add_weight(
        f'{self.name}/loc_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.loc_bias = self.add_weight(
        f'{self.name}/loc_bias',
        shape=(1, self.event_dim))
    self.scale_diag_weight = self.add_weight(
        f'{self.name}/scale_diag_weight',
        shape=(int(input_shape[-1]), self.event_dim))
    self.scale_diag_bias = self.add_weight(
        f'{self.name}/scale_diag_bias',
        shape=(1, self.event_dim))
    return

  def call(self, inputs) -> tf.Tensor:
    result = (
        tf.matmul(inputs, self.loc_weight) + self.loc_bias,
        tf.math.softplus(tf.matmul(inputs, self.scale_diag_weight) + self.scale_diag_bias)
    )
    return tf.concat(result, axis=-1)
