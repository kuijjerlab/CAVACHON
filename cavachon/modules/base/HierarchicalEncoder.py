from typing import Optional

import tensorflow as tf

class HierarchicalEncoder(tf.keras.Model):
  def __init__(self, inputs, outputs, **kwargs):
    super().__init__(inputs=inputs, outputs=outputs, **kwargs)

  @classmethod
  def make(
      cls,
      n_latent_dims: int = 5,
      z_hat_conditional_dims: Optional[int] = None,
      **kwargs) -> tf.keras.Model:
    
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