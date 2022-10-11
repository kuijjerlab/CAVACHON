from cavachon.environment.Constants import Constants
from cavachon.layers.ProgressiveScaler import ProgressiveScaler
from typing import Optional

import tensorflow as tf

class HierarchicalEncoder(tf.keras.Model):
  """HierarchicalEncoder

  HierarchicalEncoder used to encode z_hat hierarchically through the 
  dependency between components. It expects multiple tf.Tensor as 
  inputs. The key of the inputs are 'z', 'z_conditional' (if 
  applicable) and 'z_hat_conditional' (if applicable). This base module 
  is implemented using Tensorflow sequential API.

  """
  def __init__(
      self,
      n_latent_dims: int = 5,
      is_conditioned_on_z: bool = False,
      is_conditioned_on_z_hat: bool = False,
      progressive_iterations: int = 5000,
      name: str = 'hierarchical_encoder',
      **kwargs):
    """Constructor for HierarchicalEncoder. Should not be called 
    directly most of the time. Please use make() to create the model.

    Parameters
    ----------
    n_latent_dims: int, optional
        number of latent dimensions for the input z. Defaults to 5.
    
    is_conditioned_on_z: bool, optional
        use latent representation from the conditioned components.
        Defaults to False.

    is_conditioned_on_z_hat: bool, optional
        use transformed latent representation (contains information of 
        all ancestor of conditioned components) from the conditioned 
        components. Defaults to False.
    
    progressive_iterations: int, optional
        total iterations for progressive training. Defaults to 5000.

    name: str, optional:
        Name for the tensorflow model. Defaults to 
        'hierarchical_encoder'.
    """
    super().__init__(name=name)
    self.is_conditioned_on_z = is_conditioned_on_z
    self.is_conditioned_on_z_hat = is_conditioned_on_z_hat
    self.progressive_scaler = ProgressiveScaler(progressive_iterations)
    self.r_network = tf.keras.Sequential(
        [tf.keras.layers.Dense(n_latent_dims)],
        name=Constants.MODULE_R_NETWORK)
    self.b_network = tf.keras.Sequential(
        [tf.keras.layers.Dense(n_latent_dims)],
        name=Constants.MODULE_B_NETWORK)
  
  def call(
      self,
      inputs: tf.Tensor,
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Forward pass for HierarchicalEncoder.

    Parameters
    ----------
    inputs: Mapping[str, tf.Tensor]
        inputs Tensors for the HierarchicalEncoder, where keys are 'z', 
        'z_conditional' (if applicable) and 'z_hat_conditional' (if 
        applicable). (if applicable). (by defaults, expect the outputs
        by HierarchicalEncoder)

    training: bool, optional
        whether to run the network in training mode. Defaults to False.
    
    mask: tf.Tensor, optional 
        a mask or list of masks. Defaults to None.

    Returns
    -------
    tf.Tensor 
        z_hat (contains information of latent representation and the 
        conditioned components)
    
    """
    z_hat = self.r_network(inputs.get(Constants.MODEL_OUTPUTS_Z))
    concat_inputs = []
    if self.is_conditioned_on_z or self.is_conditioned_on_z_hat:
      z_hat = self.progressive_scaler(z_hat)
      if self.is_conditioned_on_z:
        concat_inputs.append(inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z, None))
      if self.is_conditioned_on_z_hat:
        concat_inputs.append(inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None))

    concat_inputs.append(z_hat)
    z_hat = tf.concat(concat_inputs, axis=-1)
    z_hat = self.b_network(z_hat)
    
    return z_hat

  def train_step(self, data):
    raise NotImplementedError()