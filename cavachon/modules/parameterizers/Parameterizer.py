from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Mapping

import tensorflow as tf

class Parameterizer(tf.keras.Model):

  default_libsize_scaling = False

  def __init__(
      self,
      inputs: Mapping[str, tf.Tensor],
      outputs: tf.Tensor,
      name: str = 'parameterizer',
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs):
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.libsize_scaling = libsize_scaling
    self.exp_transform = exp_transform
  
  @classmethod
  def setup_inputs(
      cls,
      input_dims: int,
      libsize_scaling: bool = False,
      **kwargs) -> Mapping[str, tf.keras.Input]:
    inputs = dict()
    inputs.setdefault('input', tf.keras.Input(shape=(input_dims, )))
    if libsize_scaling:
      inputs.setdefault('libsize', tf.keras.Input(shape=(1, )))
    
    return inputs

  @classmethod
  def modify_outputs(
      cls,
      inputs: Mapping[str, tf.keras.Input],
      outputs: tf.Tensor, 
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs) -> tf.Tensor:
    if libsize_scaling:
      outputs *= inputs.get('libsize')
    if exp_transform:
      outputs = tf.math.exp(outputs)
    
    return outputs

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'parameterizer',
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs):
    
    inputs = cls.setup_inputs(input_dims, libsize_scaling, **kwargs)
    layer_class = ReflectionHandler.get_class_by_name(cls.__name__, 'layers/parameterizers')
    layer = layer_class(event_dims=event_dims, name='parameterizer', **kwargs)
    outputs = layer(inputs.get('input'))
    outputs = cls.modify_outputs(
        inputs=inputs,
        outputs=outputs,
        libsize_scaling=libsize_scaling,
        exp_transform=exp_transform,
        **kwargs)

    return cls(
        inputs=inputs,
        outputs=outputs,
        name=name,
        libsize_scaling=libsize_scaling,
        exp_transform=exp_transform)

