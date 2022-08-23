import tensorflow as tf

class MultivariateNormalDiag(tf.keras.layers.Layer):
  def __init__(
      self,
      event_dims: int,
      name: str = 'MultivariateNormalDiagParameterizer'):
    super().__init__(name=name)
    self.event_dims: int = event_dims
    return

  def build(self, input_shape: tf.TensorShape) -> None:
    self.loc_weight = self.add_weight(
        f'{self.name}/loc_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.loc_bias = self.add_weight(
        f'{self.name}/loc_bias',
        shape=(1, self.event_dims))
    self.scale_diag_weight = self.add_weight(
        f'{self.name}/scale_diag_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.scale_diag_bias = self.add_weight(
        f'{self.name}/scale_diag_bias',
        shape=(1, self.event_dims))
    return

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    result = (
        tf.matmul(inputs, self.loc_weight) + self.loc_bias,
        tf.math.softplus(tf.matmul(inputs, self.scale_diag_weight) + self.scale_diag_bias)
    )
    # shape: (batch, event_dims * 2)
    return tf.concat(result, axis=-1)
