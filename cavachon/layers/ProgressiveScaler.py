#%%
import tensorflow as tf

class ProgressiveScaler(tf.keras.layers.Layer):
  """ProgressiveScaler

  ProgressiveScaler used to scale the inputs during training. The input 
  tensor will be scale as current_iteration/iteration * tensor. Do 
  nothing in inference mode.

  Attributes
  ----------
  total_iterations: tf.Variable
      total iterations in the progressive training.

  current_iteration: tf.Variable
      current iterations in the progressive training.

  """
  def __init__(self, total_iterations: int = 5000, name: str = 'progressive_scaler'):
    """Constructor for ProgressiveScaler

    Parameters
    ----------
    total_iterations: int, optional
        total iterations in the progressive training. Defaults to 5000.
    
    name: str, optional
        Name for the tensorflow layer. Defaults to 'progressive_scaler'.

    """
    super().__init__(name=name)
    self.total_iterations = tf.Variable(total_iterations, trainable=False, dtype=tf.float32)
    self.current_iteration = tf.Variable(tf.ones(()), trainable=False)
  
  def call(
      self,
      inputs: tf.Tensor,
      training: bool = False,
      **kwargs) -> tf.Tensor:
    """Forward pass for ProgressiveScaler

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor for the encoder, expect a single Tensor (by 
        defaults, z_hat of conditioned component)

    training: bool, optional
        whether to run the network in training mode. Defaults to False.
    
    mask: tf.Tensor, optional 
        a mask or list of masks. Defaults to None.

    Returns
    -------
    tf.Tensor
        parameters for the latent distributions.
    
    """
    if training:
        alpha = (self.current_iteration + 1e-7) / (self.total_iterations + 1e-7)
        alpha = tf.where(alpha > 1.0, tf.ones_like(alpha), alpha)
        result = alpha * inputs
        self.current_iteration.assign_add(1.0)
        self.current_iteration.assign(
            tf.where(
              self.current_iteration > self.total_iterations,
              self.total_iterations,
              self.current_iteration))
    else:
        result = inputs

    return result
