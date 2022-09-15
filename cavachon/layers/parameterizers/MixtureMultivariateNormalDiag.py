#%%
import tensorflow as tf

class MixtureMultivariateNormalDiag(tf.keras.layers.Layer):
  """MixtureMultivariateNormalDiag
  
  Parameterizer for mixture of multivariate normal distributions with 
  diagonal covariance matrix (logits, loc and scale_diag).

  """
  def __init__(
      self,
      event_dims: int,
      n_components: int,
      unit_variance: bool = True,
      name: str = 'mixture_multivariate_normal_diag_parameterizer'):
    """Constructor for MultivariateNormalDiag

    Parameters
    ----------
    event_dims: int
        number of event dimensions for the multivariate normal 
        distributions with diagonal covariance matrix.

    n_components: int
        number of components in the mixture distributions.

    unit_variance: bool, optional
        use unit variance. Defaults to True.

    name: str, optional
        Name for the tensorflow layer. Defaults to 
        'mixture_multivariate_normal_diag_parameterizer'.

    """
    super().__init__(name=name)
    self.event_dims: int = event_dims
    self.n_components: int = n_components
    self.unit_variance: bool = unit_variance
    return

  def build(self, input_shape: tf.TensorShape) -> None:
    """Create necessary tf.Variable for the first time being called.
    (see tf.keras.layers.Layer) 

    Parameters
    ----------
    input_shape: tf.TensorShape
        input shape of tf.Tensor.

    """
    self.logits_weight = self.add_weight(
        f'{self.name}/logits_weight',
        shape=(int(input_shape[-1]), self.n_components),
        initializer=tf.keras.initializers.Constant(0.))
    self.logits_bias = self.add_weight(
        f'{self.name}/logits_bias',
        shape=(1, self.n_components),
        initializer=tf.keras.initializers.Constant(0.))
    
    self.loc_weight = []
    self.loc_bias = []
    if not self.unit_variance:
      self.scale_diag_weight = []
      self.scale_diag_bias = []
    
    for i in range(self.n_components):
      self.loc_weight.append(self.add_weight(
          f'{self.name}/loc_weight_{i}',
          shape=(int(input_shape[-1]), self.event_dims)))
      self.loc_bias.append(self.add_weight(
          f'{self.name}/loc_bias_{i}',
          shape=(1, self.event_dims)))
      if not self.unit_variance:
        self.scale_diag_weight.append(self.add_weight(
            f'{self.name}/scale_diag_weight_{i}',
            shape=(int(input_shape[-1]), self.event_dims)))
        self.scale_diag_bias.append(self.add_weight(
            f'{self.name}/scale_diag_bias_{i}',
            shape=(1, self.event_dims)))

    return

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    """Parameterize mixture of multivariate normal distributions with 
    diagonal covariance matrix with loc and scale_diag using the given 
    tf.Tensor.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor.

    Returns
    -------
    tf.Tensor
        logits, loc and scale_diag for normal distributions with 
        diagonal covariance matrix, with shape 
        (batch, n_components, event_dims * 2 + 1), where
        1. results[..., n_components, 0] are the logits
        2. results[..., n_components, 1:event_dims+1] are the loc 
           (mean)
        3. results[..., n_components, event_dims+1:] are the scale_diag 
           (std)

    """

    # shape: (batch, n_components, 1)
    logits = tf.expand_dims(tf.matmul(inputs, self.logits_weight) + self.logits_bias, -1)
    means = []
    scale_diag = []
    for i in range(self.n_components):
      loc_weight = self.loc_weight[i]
      loc_bias = self.loc_bias[i]
      if not self.unit_variance:
        scale_diag_weight = self.scale_diag_weight[i]
        scale_diag_bias = self.scale_diag_bias[i]
      
      mean =  tf.matmul(inputs, loc_weight) + loc_bias

      means.append(mean),
      if not self.unit_variance:
        scale_diag.append(
            tf.math.softplus(tf.matmul(inputs, scale_diag_weight) + scale_diag_bias) + 1e-7)
      else:
        scale_diag.append(tf.ones_like(mean))

    # shape: (batch, n_components, event_dims)
    means = tf.stack(means, axis=1)
    scale_diag = tf.stack(scale_diag, axis=1)

    # shape: (batch, n_components, event_dims * 2 + 1)
    return tf.concat([logits, means, scale_diag], axis=-1)
