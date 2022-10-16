import tensorflow as tf

class MultivariateNormalDiag(tf.keras.layers.Layer):
  """MultivariateNormalDiag
  
  Parameterizer for multivariate normal distributions with diagonal 
  covariance matrix (loc and scale_diag).

  """
  def __init__(
      self,
      event_dims: int,
      name: str = 'multivariate_normal_diag_parameterizer'):
    """Constructor for MultivariateNormalDiag

    Parameters
    ----------
    event_dims: int
        number of event dimensions for the multivariate normal 
        distributions with diagonal covariance matrix.
        
    name: str, optional
        Name for the tensorflow layer. Defaults to 
        'multivariate_normal_diag_parameterizer'.
    """
    super().__init__(name=name)
    self.event_dims: int = event_dims
    return

  def build(self, input_shape: tf.TensorShape) -> None:
    """Create necessary tf.Variable for the first time being called.
    (see tf.keras.layers.Layer) 

    Parameters
    ----------
    input_shape: tf.TensorShape
        input shape of tf.Tensor.

    """
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

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    """Parameterize multivariate normal distributions with diagonal 
    covariance matrix with loc and scale_diag using the given tf.Tensor.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor.

    Returns
    -------
    tf.Tensor
        loc and scale_diag for normal distributions with diagonal 
        covariance matrix, with shape (batch, event_dims * 2), where 
        1. results[..., 0:event_dims] are the loc (mean)
        2. results[..., event_dims:2*event_dims] are the scale_diag 
           (std)

    """
    mean = tf.matmul(inputs, self.loc_weight) + self.loc_bias
    result = (
        mean,
        tf.math.softplus(tf.matmul(inputs, self.scale_diag_weight) + self.scale_diag_bias) + 1e-7
    )
    # shape: (batch, event_dims * 2)
    return tf.concat(result, axis=-1)
