from cavachon.utils.ReflectionHandler import ReflectionHandler
from cavachon.utils.TensorUtils import TensorUtils
from typing import Dict, Optional

import tensorflow as tf 

class DecoderDataParameterizer(tf.keras.Model):
  """DecoderDataParameterizer

  Decoder and parameterizer for data distributions. This base module is 
  implemented with Tensorflow sequential API.

  Attributes
  ----------
  backbone_network: tf.keras.Model
      backbone network for decoder.
  
  x_parameterizer: tf.keras.Model
      parameterizer for data distribution.
      
  """
  def __init__(
      self,
      distribution_name: str,
      n_vars: int = 5,
      n_layers: int = 3,
      *args,
      **kwargs):
    """Constructor DecoderDataParameterizer

    Parameters
    ----------
    distribution_name: str
        distribution name for the modality to be decoded. By default,
        this is used to automatically find the parameterizer in
        modules.parameterizers

    n_vars: int, optional:
        number of event dimension for data distributions. Defaults to 5.
    
    n_layers: int, optional
        number of layers constructed in the backbone network. Defaults 
        to 3.

    """
    super().__init__(self, *args, **kwargs)
    
    distribution_parameterizer = ReflectionHandler.get_class_by_name(
        distribution_name,
        'modules/parameterizers')

    self.backbone_network = TensorUtils.create_backbone_layers(
        n_layers,
        name='backbone_network')

    input_dims = 0
    for layer in self.backbone_network.layers[::-1]:
      if isinstance(layer, tf.keras.layers.Dense):
        input_dims = layer.units
        break
    
    self.x_parameterizer = distribution_parameterizer.make(
        input_dims = input_dims,
        event_dims = n_vars,
        name='x_parameterizer')

  def call(
      self,
      inputs: tf.Tensor,
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Forward pass for DecoderDataParameterizer.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor for the decoder, expect a single Tensor (by 
        defaults, the z_hat outputs by HierarchicalEncoder)

    training: bool, optional
        whether to run the network in training mode. Defaults to False.
    
    mask: tf.Tensor, optional 
        a mask or list of masks. Defaults to None.

    Returns
    -------
    tf.Tensor
        parameters for the data distributions.
    
    """
    result = self.backbone_network(inputs.get('input'), training=training, mask=mask)
    x_parameterizer_inputs = dict()
    x_parameterizer_inputs.setdefault('input', result)
    if self.x_parameterizer.libsize_scaling:
      x_parameterizer_inputs.setdefault('libsize', inputs.get('libsize'))
    result = self.x_parameterizer(x_parameterizer_inputs, training=training, mask=mask)

    return result

  def train_step(self, data):
    raise NotImplementedError()