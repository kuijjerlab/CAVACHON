from cavachon.environment.Constants import Constants
from cavachon.modules.conditionals.ConditionalModule import ConditionalModule
from cavachon.utils.TensorUtils import TensorUtils
from typing import Any, Collection, Dict, Union, List, Mapping, MutableMapping, Optional

import tensorflow as tf

class Independent(ConditionalModule):
  def __init__(
      self,
      order: int,
      modality_names: Collection[str] = [],
      distribution_names: List[Union[str, str]] = dict(), 
      n_vars: Mapping[str, int] = dict(),
      n_latent_dims: Mapping[str, int] = dict(),
      n_priors: Mapping[str, int] = dict(),           
      n_encoder_layers: Mapping[str, int] = dict(),  
      n_decoder_layers: Mapping[str, int] = dict(),   
      conditional_module_name: Optional[str] = None,      
      name: str = 'IndependentModule'):

    super().__init__(
      order=order,
      modality_names=modality_names,
      distribution_names=distribution_names,
      n_vars=n_vars,
      n_latent_dims=n_latent_dims,
      n_priors=n_priors, 
      n_encoder_layers=n_encoder_layers,
      n_decoder_layers=n_decoder_layers, 
      conditional_module_name=conditional_module_name,
      name=name
    )

  def setup_encoder(self) -> None:
    super().setup_encoder()
    self.input_layers: Dict[str, tf.keras.layers.Layer] = dict()
    for modality_name in self.modality_names:
      backbone_network = self.encoder_backbone_networks.get(modality_name)
      max_n_neurons = TensorUtils.max_n_neurons(backbone_network.layers)
      self.input_layers.setdefault(modality_name, tf.keras.layers.Dense(max_n_neurons))

  def call(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    
    for modality_name in self.modality_names:
      input_layer = self.input_layers.get(modality_name)
      matrix = inputs.get((modality_name, Constants.TENSOR_NAME_X))
      if len(matrix.shape) == 1:
        matrix = tf.reshape(matrix, (1, -1))
      inputs[(modality_name, Constants.TENSOR_NAME_X)] = input_layer(matrix)
    
    return super().call(inputs, training, mask)