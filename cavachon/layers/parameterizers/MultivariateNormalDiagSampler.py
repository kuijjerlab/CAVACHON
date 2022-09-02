import tensorflow as tf

class MultivariateNormalDiagSampler(tf.keras.layers.Layer):
  """MultivariateNormalDiagSampler

  Sampler for MultivariateNormalDiag from the outputs of Parameterizer

  """
  def __init__(
      self,
      name: str = 'multivariate_normal_diag_sampler'):
    """Constructor for MultivariateNormalDiagSampler

    Parameters
    ----------
    name: str, optional
        Name for the tensorflow layer. Defaults to 
        'multivariate_normal_diag_sampler'.
    """
    super().__init__(name=name)
    return

  def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
    """Sample for MultivariateNormalDiag from the outputs of 
    Parameterizer.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor.
    
    training: bool, optional
        if the layer is called in training mode. Defaults to False.

    Returns
    -------
    tf.Tensor
        sampling results.
    """
    loc, scale_diag = tf.split(inputs, 2, axis=-1)
    if training:
      epsilon = tf.random.normal(shape=tf.shape(loc))
      return epsilon * scale_diag + loc
    else:
      return loc
