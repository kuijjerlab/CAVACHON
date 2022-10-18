from cavachon.distributions.Distribution import Distribution
import tensorflow as tf

class IndependentZeroInflatedNegativeBinomial(tf.keras.layers.Layer):
  """IndependentZeroInflatedNegativeBinomial
  
  Parameterizer for IndependentZeroInflatedNegativeBinomial 
  distributions (logits, mean and dispersion).

  """
  def __init__(
      self,
      event_dims: int,
      use_shared_dispersion: bool = True,
      name: str = 'independent_zero_inflated_negative_binomial_parameterizer'):
    """Constructor for IndependentZeroInflatedNegativeBinomial

    Parameters
    ----------
    event_dims: int
        number of event dimensions for the independent zero-inflated
        negative binomial distribution.
    
    use_shared_dispersion: bool
        use shared dispersion across all samples instead of modeling 
        the dispersions independently for each sample (see Rybkin 
        et al., 2021)       
 
    name: str, optional
        Name for the tensorflow layer. Defaults to 
        'independent_zero_inflated_negative_binomial_parameterizer'.
    """
    super().__init__(name=name)
    self.event_dims: int = event_dims
    self.use_shared_dispersion: bool = use_shared_dispersion
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
    if self.use_shared_dispersion:
      self.dispersion_weight = self.add_weight(
          f'{self.name}/dispersion_weight',
          shape=(1, self.event_dims))
    else:
      self.dispersion_weight = self.add_weight(
          f'{self.name}/dispersion_weight',
          shape=(int(input_shape[-1]), self.event_dims))
    self.dispersion_bias = self.add_weight(
        f'{self.name}/dispersion_bias',
        shape=(1, self.event_dims))

    return

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    """Parameterize independent zero-inflated negative binomial 
    distributions with logits, mean and dispersion using the given 
    tf.Tensor.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor.

    Returns
    -------
    tf.Tensor
        logits, mean and dispersion for zero-inflated negative binomial 
        distributions, with shape (batch, event_dims * 3), where 
        1. results[..., 0:event_dims] are the logits
        2. results[..., event_dims:2*event_dims] are the means
        3. results[..., 2*event_dims:3*event_dims] are the dispersions

    """
    if self.use_shared_dispersion:
      batch_size = tf.shape(inputs)[0]
      dispersion = tf.math.sigmoid(self.dispersion_weight + self.dispersion_bias)
      dispersion = tf.repeat(dispersion, repeats=batch_size, axis=0)
    else:
      dispersion = tf.math.sigmoid(tf.matmul(inputs, self.dispersion_weight) + self.dispersion_bias)
    dispersion = tf.where(dispersion <= 0.1, 0.1 * tf.ones_like(dispersion), dispersion)
    
    result = (
        tf.matmul(inputs, self.logits_weight) + self.logits_bias,
        tf.math.softmax(tf.matmul(inputs, self.mean_weight) + self.mean_bias),
        dispersion,
    )
    # shape: (batch, event_dims * 3)
    return tf.concat(result, axis=-1)