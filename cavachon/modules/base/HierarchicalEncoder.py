from typing import Any, Mapping, Optional

import tensorflow as tf

class HierarchicalEncoder(tf.keras.Model):
  """HierarchicalEncoder

  HierarchicalEncoder used to encode z_hat hierarchically through the 
  dependency between components. It expects multiple tf.Tensor as 
  inputs ('z' and 'z_hat_conditional') as inputs. This base module is 
  implemented using Tensorflow functional API.

  """
  def __init__(
      self,
      inputs: Mapping[Any, tf.keras.Input],
      outputs: tf.Tensor,
      name: str = 'hierarchical_encoder',
      **kwargs):
    """Constructor for HierarchicalEncoder. Should not be called 
    directly most of the time. Please use make() to create the model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]): 
        inputs for building tf.keras.Model using Tensorflow functional 
        API. Expect to have key 'z' and 'z_hat_conditional' by defaults.
    
    outputs: tf.Tensor:
        outputs z_hat for building tf.keras.Model using Tensorflow 
        functional API.
    
    name: str, optional:
        Name for the tensorflow model. Defaults to 'preprocessor'.
    """
    super().__init__(inputs=inputs, outputs=outputs, name=name)

  @classmethod
  def make(
      cls,
      n_latent_dims: int = 5,
      z_hat_conditional_dims: Optional[int] = None,
      **kwargs) -> tf.keras.Model:
    """Make the tf.keras.Model using the functional API of Tensorflow.

    Parameters
    ----------
    n_latent_dims: int, optional
        number of latent dimensions for the input z. Defaults to 5.
    
    z_hat_conditional_dims: int, optional
        dimension of z_hat from the components of the dependency. None 
        if the component does not depends on any other components. 
        Defaults to None.

    Returns
    -------
    tf.keras.Model
        created model using Tensorflow functional API.

    """
    inputs = dict()
    inputs.setdefault('z', tf.keras.Input(shape=(n_latent_dims, ), name='z'))
    if z_hat_conditional_dims is not None:
      inputs.setdefault(
          'z_hat_conditional', 
          tf.keras.Input(shape=(z_hat_conditional_dims, ), name='z_hat_conditional'))

    r_network = tf.keras.Sequential(
        [tf.keras.layers.Dense(n_latent_dims)],
        name='r_network')
    b_network = tf.keras.Sequential(
        [tf.keras.layers.Dense(n_latent_dims)],
        name='b_network')
    
    z_hat = r_network(inputs.get('z'))
    z_hat_conditional = inputs.get('z_hat', None)
    if z_hat_conditional is None:
        z_hat = b_network(z_hat)
    else:
        z_hat = tf.concat([z_hat_conditional, z_hat], axis=-1)
        z_hat = b_network(z_hat)

    return cls(inputs=inputs, outputs=z_hat, **kwargs)

  def train_step(self, data):
    raise NotImplementedError()