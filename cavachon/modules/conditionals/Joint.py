from cavachon.environment.Constants import Constants
from cavachon.modules.conditionals.ConditionalModule import ConditionalModule
from cavachon.utils.TensorUtils import TensorUtils
from typing import Any, Collection, Dict, Union, List, Mapping, MutableMapping, Optional

import tensorflow as tf

class Joint(ConditionalModule):
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
      name: str = 'JointModule'):

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
  
  def setup_and_validate_parameters(self, *args, **kwargs) -> None:
    self.latent_names = (f"Joint{tuple(self.modality_names)}", )
    self.n_encoders = len(self.latent_names)
    self.latent_name = list(self.latent_names).pop()
    super().setup_and_validate_parameters(*args, **kwargs)

  def setup_encoder(self) -> None:
    super().setup_encoder()
    self.input_layers: Dict[str, tf.keras.layers.Layer] = dict()
    backbone_network = self.encoder_backbone_networks.get(self.latent_name)
    max_n_neurons = TensorUtils.max_n_neurons(backbone_network.layers)
    for modality_name in self.modality_names:
      self.input_layers.setdefault(modality_name, tf.keras.layers.Dense(max_n_neurons))
    
  def encode_hierarchically(self, *args, **kwargs) -> Dict[Any, tf.Tensor]:
    z_sampled = super().encode_hierarchically(*args, **kwargs)
    z_hat = z_sampled.get((f"{self.order:02d}", self.name, 'z_hat'))
    for modality_name in self.modality_names:
      z_sampled.setdefault((f"{self.order:02d}", self.name, modality_name, 'z_hat'), z_hat)
    return z_sampled

  def call(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    
    result = []
    for modality_name in self.modality_names:
      input_layer = self.input_layers.get(modality_name)
      matrix = inputs.get((modality_name, Constants.TENSOR_NAME_X))
      if len(matrix.shape) == 1:
        matrix = tf.reshape(matrix, (1, -1))
      result.append(input_layer(matrix))
    
    inputs[(self.latent_name, Constants.TENSOR_NAME_X)] = tf.concat(result, axis=-1)
    
    return super().call(inputs, training, mask)