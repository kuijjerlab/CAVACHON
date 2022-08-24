from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag as MultivariateNormalDiagDistribution
from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from cavachon.utils.TensorUtils import TensorUtils
from cavachon.layers.parameterizers.MixtureMultivariateNormalDiag import MixtureMultivariateNormalDiag as MixtureMultivariateNormalDiagParameterizer
from cavachon.layers.parameterizers.MultivariateNormalDiag import MultivariateNormalDiag as MultivariateNormalDiagParameterizer
from typing import Any, Collection, Dict, Union, List, Mapping, MutableMapping, Optional

import tensorflow as tf

class ConditionalModule(tf.keras.Model):
  def __init__(
      self,
      order: int,
      modality_names: Collection[str] = [],
      dist_parameterizers: List[Union[str, str]] = dict(), 
      n_vars: Mapping[str, int] = dict(),
      n_latent_dims: Mapping[str, int] = dict(),     
      n_priors: Mapping[str, int] = dict(),          
      n_encoder_layers: Mapping[str, int] = dict(), 
      n_decoder_layers: Mapping[str, int] = dict(),  
      conditional_module_name: Optional[str] = None,
      name: str = 'Module'):

    super().__init__(name=name)

    self.order: int = order
 
    self.latent_names: Collection[Any] = modality_names
    self.modality_names: Collection[Any] = modality_names

    self.n_encoders: int = len(self.latent_names)
    self.n_decoders: int = len(self.modality_names)

    self.n_vars: Mapping[str, int] = n_vars
    self.n_encoder_layers: Mapping[str, int] = n_encoder_layers
    self.n_decoder_layers: Mapping[str, int] = n_decoder_layers
    self.n_latent_dims: Mapping[str, int] = n_latent_dims
    self.n_priors: Mapping[str, int] = n_priors
    self.conditional_module_name: Optional[Any] = conditional_module_name
    self.dist_parameterizers: Mapping[str, str] = dist_parameterizers
  
    self.setup_and_validate_parameters()
    self.setup_encoder()
    self.setup_decoder()

  def setup_and_validate_parameters(
      self,
      default_n_latent_dims: int = 6,
      default_n_priors: int = 13,
      default_n_encoder_layers: int = 3,
      default_n_decoder_layers: int = 3) -> None:
    
    for encoder_key in self.latent_names:
      self.n_latent_dims.setdefault(encoder_key, default_n_latent_dims)
      self.n_priors.setdefault(encoder_key, default_n_priors)
      self.n_decoder_layers.setdefault(encoder_key, default_n_encoder_layers)

    for decoder_key in self.modality_names:
      self.n_decoder_layers.setdefault(decoder_key, default_n_decoder_layers)

      if decoder_key not in self.dist_parameterizers:
        message = ''.join((
            f"'{decoder_key}' not in distribution_parameterizers ",
            f"({self.__class__.__name__}.{self.name})"
        ))
        raise KeyError(message)

      if decoder_key not in self.n_vars:
        message = ''.join((
            f"'{decoder_key}' not in n_vars ",
            f"({self.__class__.__name__}.{self.name})"
        ))
        raise KeyError(message)

    return 

  def setup_encoder(self) -> None:

    self.encoder_backbone_networks: Dict[str, tf.keras.Model] = dict()
    self.z_parameterizers: Dict[str, MultivariateNormalDiagParameterizer] = dict()
    self.z_prior_parameterizers: Dict[str, MixtureMultivariateNormalDiagParameterizer] = dict()
    self.r_networks: Dict[str, tf.keras.Model] = dict()
    self.b_networks: Dict[str, tf.keras.Model] = dict()
  
    for latent_name in self.latent_names:
      n_latent_dims = self.n_latent_dims.get(latent_name)
      n_priors = self.n_priors.get(latent_name)
      n_encoder_layers = self.n_decoder_layers.get(latent_name)
      
      name_prefix = f"{self.name}/{latent_name}/"

      backbone_network = TensorUtils.create_backbone_layers(
          n_encoder_layers,
          name=f"{name_prefix}/backbone_network")
      z_parameterizer = MultivariateNormalDiagParameterizer(
          n_latent_dims,
          name=f"{name_prefix}/z_parameterizer")
      z_prior_parameterizer = MixtureMultivariateNormalDiagParameterizer(
          n_latent_dims,
          n_priors,
          name=f"{name_prefix}/z_prior_parameterizer")
      r_network = tf.keras.Sequential(
          [tf.keras.layers.Dense(n_latent_dims)],
          name=f"{name_prefix}/r_network")
      b_network = tf.keras.Sequential(
          [tf.keras.layers.Dense(n_latent_dims)],
          name=f"{name_prefix}/b_network")
      
      self.encoder_backbone_networks.setdefault(latent_name, backbone_network)
      self.z_parameterizers.setdefault(latent_name, z_parameterizer)
      self.z_prior_parameterizers.setdefault(latent_name, z_prior_parameterizer)
      self.r_networks.setdefault(latent_name, r_network)
      self.b_networks.setdefault(latent_name, b_network)
    
    return

  def setup_decoder(self) -> None:
    self.decoder_backbone_networks: Dict[str, tf.keras.Model] = dict()
    self.x_parameterizers: Dict[str, tf.keras.Model] = dict()
    for decoder_name in self.modality_names:
      distribution_parameterizer = self.dist_parameterizers.get(decoder_name)
      n_vars = self.n_vars.get(decoder_name)
      n_decoder_layers = self.n_decoder_layers.get(decoder_name)

      name_prefix = f"{self.name}/{decoder_name}/"

      backbone_network = TensorUtils.create_backbone_layers(
          n_decoder_layers,
          reverse=True,
          name=f"{name_prefix}/backbone_network")
      
      if isinstance(distribution_parameterizer, str):
        distribution_parameterizer = ReflectionHandler.get_class_by_name(
            distribution_parameterizer,
            'layers/parameterizers')
      
      x_parameterizer = distribution_parameterizer(
          n_vars,
          name=f"{name_prefix}/x_parameterizer")
    
      self.decoder_backbone_networks.setdefault(decoder_name, backbone_network)
      self.x_parameterizers.setdefault(decoder_name, x_parameterizer)

  def call(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    
    z_hat_conditional = inputs.get((self.conditional_module_name, 'z_hat'), None)

    z_parameters = self.encode_z_parameters(inputs, training=training)
    inputs.update(z_parameters)
    z_sampled = self.encode_hierarchically(z_parameters, z_hat_conditional, training=training)
    inputs.update(z_sampled)
    x_parameters = self.decode(inputs, training=training)
    inputs.update(x_parameters)

    for latent_name in self.latent_names:
      inputs.pop((latent_name, Constants.TENSOR_NAME_X), None)
    for modality_name in self.modality_names:
      inputs.pop((modality_name, Constants.TENSOR_NAME_X), None)

    return inputs

  def encode_z_parameters(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False) -> Dict[str, tf.Tensor]:
    
    z_parameters = dict()
    for latent_name in self.latent_names:
      backbone_network = self.encoder_backbone_networks.get(latent_name)
      z_parameterizer = self.z_parameterizers.get(latent_name)
      matrix = inputs.get((latent_name, Constants.TENSOR_NAME_X))
      if len(matrix.shape) == 1:
        matrix = tf.reshape(matrix, (1, -1))
      x_encoded = backbone_network(matrix, training=training)
      z_parameters.setdefault(
          (f"{self.order:02d}", self.name, latent_name , 'z_parameters'),
          z_parameterizer(x_encoded))

    return z_parameters
    
  def encode_hierarchically(
      self,
      z_parameters: Dict[str, tf.Tensor],
      z_hat_conditional: Optional[tf.Tensor],
      training: bool = False) -> Dict[Any, tf.Tensor]:
    
    z_sampled = dict()
    z_hat_list = list()
    for latent_name in self.latent_names:
      z_parameter = z_parameters.get(
          (f"{self.order:02d}", self.name, latent_name, 'z_parameters'))
      # eq D.63 (Falck et al., 2021)
      z = MultivariateNormalDiagDistribution.from_parameterizer_output(z_parameter).sample()
      if z_hat_conditional is None:
        # eq D.64 (Falck et al., 2021)
        z_hat = self.r_networks.get(latent_name)(z, training=training)
        z_hat = self.b_networks.get(latent_name)(z_hat, training=training)
      else:
        # eq D.65 (Falck et al., 2021)
        z_hat = self.r_networks.get(latent_name)(z, training=training)
        z_hat = tf.concat([z_hat_conditional, z_hat], axis=-1)
        z_hat = self.b_networks.get(latent_name)(z_hat, training=training)
      
      z_hat_list.append(z_hat)
      z_sampled.setdefault((f"{self.order:02d}", self.name, latent_name, 'z'), z)
      z_sampled.setdefault((f"{self.order:02d}", self.name, latent_name, 'z_hat'), z_hat)

    z_sampled.setdefault(
        (f"{self.order:02d}", self.name, 'z_hat'),
        tf.concat(z_hat_list, axis=-1))
    
    return z_sampled

  def decode(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False) -> Dict[str, tf.Tensor]:
    x_parameters = dict()
    for decoder_name in self.modality_names:
      z_hat = inputs.get((f"{self.order:02d}", self.name, decoder_name, 'z_hat'))

      batch_effect = inputs.get((decoder_name, Constants.TENSOR_NAME_BATCH), None)
      if batch_effect and len(batch_effect.shape) == 1:
        batch_effect = tf.reshape(batch_effect, (-1, 1))
      
      decoder_input = z_hat
      if batch_effect is not None:
        decoder_input = tf.concat([decoder_input, batch_effect], axis=-1)
      
      backbone_network = self.decoder_backbone_networks.get(decoder_name)
      x_parameterizer = self.x_parameterizers.get(decoder_name)
      z_decoded = backbone_network(decoder_input, training=training)
      x_parameters.setdefault(
          (f"{self.order:02d}", self.name, decoder_name, 'x_parameters'),\
          x_parameterizer(z_decoded))
    
    return x_parameters
