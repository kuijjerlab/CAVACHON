from cavachon.environment.Constants import Constants
from cavachon.layers.parameterizers.MultivariateNormalDiag import MultivariateNormalDiag as MultivariateNormalDiagParameterizer
from cavachon.utils.TensorUtils import TensorUtils
from typing import Dict, Optional

import tensorflow as tf 

class EncoderLatentParameterizer(tf.keras.Model):
  """EncoderLatentParameterizer
  
  Encoder and parameterizer for latent distributions. This base module 
  is implemented using Tensorflow sequential API.

  Attributes
  ----------
  backbone_network: tf.keras.Model
      backbone network for encoder.
  
  z_parameterizer: tf.keras.Model
      parameterizer for latent distribution.

  max_n_neurons: int
      maximum number of neurons in the backbone network (used to 
      initialize modules.preprocessors.Preprocessor)

  """
  def __init__(
      self,
      n_layers: int = 3,
      n_latent_dims: int = 5,
      *args,
      **kwargs):
    super().__init__(self, *args, **kwargs)
    self.backbone_network = TensorUtils.create_backbone_layers(
        n_layers,
        reverse=True,
        name=Constants.MODULE_BACKBONE)
    self.z_parameterizer = MultivariateNormalDiagParameterizer(
        n_latent_dims,
        name=Constants.MODULE_Z_PARAMETERIZER)
    self.max_n_neurons = TensorUtils.max_n_neurons(self.backbone_network.layers)

  def call(
      self,
      inputs: tf.Tensor,
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    """Forward pass for EncoderLatentParameterizer.

    Parameters
    ----------
    inputs: tf.Tensor
        inputs Tensor for the encoder, expect a single Tensor (by 
        defaults, the outputs by Preprocessor)

    training: bool, optional
        whether to run the network in training mode. Defaults to False.
    
    mask: tf.Tensor, optional 
        a mask or list of masks. Defaults to None.

    Returns
    -------
    tf.Tensor
        parameters for the latent distributions.
    
    """

    result = self.backbone_network(inputs, training=training, mask=mask)
    result = self.z_parameterizer(result, trainin=training, mask=mask)

    return result

  def train_step(self, data):
    raise NotImplementedError()
