import tensorflow as tf

class MultivariateNormalDiagSampler(tf.keras.layers.Layer):
  def __init__(
      self,
      name: str = 'MultivariateNormalDiagSampler'):
    super().__init__(name=name)
    return

  def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
    loc, scale_diag = tf.split(inputs, 2, axis=-1)
    if training:
      epsilon = tf.random.normal(shape=tf.shape(loc))
      return epsilon * scale_diag + loc
    else:
      return loc
