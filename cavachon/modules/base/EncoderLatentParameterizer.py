from cavachon.layers.parameterizers.MultivariateNormalDiag import MultivariateNormalDiag as MultivariateNormalDiagParameterizer
from cavachon.utils.TensorUtils import TensorUtils
from typing import Dict, Optional

import tensorflow as tf 

class EncoderLatentParameterizer(tf.keras.Model):
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
        name='backbone_network')
    self.z_parameterizer = MultivariateNormalDiagParameterizer(
        n_latent_dims,
        name='z_parameterizer')
    self.max_n_neurons = TensorUtils.max_n_neurons(self.backbone_network.layers)

  def call(
    self,
    inputs: tf.Tensor,
    training: bool = False,
    mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    
    result = self.backbone_network(inputs, training=training, mask=mask)
    result = self.z_parameterizer(result, trainin=training, mask=mask)

    return result

  def train_step(self, data):
    raise NotImplementedError()
