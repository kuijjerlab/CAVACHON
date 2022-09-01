from cavachon.modules.parameterizers.Parameterizer import Parameterizer
from typing import Mapping

import tensorflow as tf

class IndependentZeroInflatedNegativeBinomial(Parameterizer):

  default_libsize_scaling = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @classmethod
  def modify_outputs(
      cls,
      inputs: Mapping[str, tf.keras.Input],
      outputs: tf.Tensor, 
      libsize_scaling: bool = True,
      exp_transform: bool = True,
      **kwargs) -> tf.Tensor:

    logits, mean, dispersion = tf.split(outputs, 3, axis=-1)
    if libsize_scaling:
      mean *= inputs.get('libsize')
    if exp_transform:
      mean = tf.where(mean > 7., 7. * tf.ones_like(mean), mean)
      mean = tf.math.exp(mean) - 1.0
    
    mean = tf.where(mean == 0, 1e-7 * tf.ones_like(mean), mean)

    return tf.concat([logits, mean, dispersion], axis=-1)

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'independent_zero_inflated_negative_binomial',
      libsize_scaling: bool = True,
      exp_transform: bool = True):

    return super().make(
      input_dims=input_dims,
      event_dims=event_dims,
      name=name,
      libsize_scaling=libsize_scaling,
      exp_transform=exp_transform)