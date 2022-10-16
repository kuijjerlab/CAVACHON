import tensorflow as tf

class IndependentBernoulli(tf.keras.layers.Layer):
  """IndependentBernoulli
  
  Parameterizer for IndependentBernoulli distributions (logits).

  """
  def __init__(
      self,
      event_dims: int,
      name: str = 'independent_bernoulli_parameterizer'):
    """Constructor for IndependentBenoulli

    Parameters
    ----------
    event_dims: int
        number of event dimensions for the independent Bernoulli 
        distribution.
        
    name: str, optional
        Name for the tensorflow layer. Defaults to 
        'independent_bernoulli_parameterizer'.
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
    self.logits_weight = self.add_weight(
        f'{self.name}/logits_weight',
        shape=(int(input_shape[-1]), self.event_dims))
    self.logits_bias = self.add_weight(
        f'{self.name}/logits_bias',
        shape=(1, self.event_dims))
    return

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    """Parameterize independent Bernoulli distributions with logits 
    using the given tf.Tensor.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor.

    Returns
    -------
    tf.Tensor
        logits for IndependentBernoulli distributions, with shape 
        (batch, event_dims)
        
    """
    # shape: (batch, event_dims)
    return tf.matmul(inputs, self.logits_weight) + self.logits_bias

