from cavachon.environment.Constants import Constants
from cavachon.layers.modifiers.ToDense import ToDense
from cavachon.layers.parameterizers.MixtureMultivariateNormalDiag import MixtureMultivariateNormalDiag
from cavachon.layers.parameterizers.MultivariateNormalDiagSampler import MultivariateNormalDiagSampler
from cavachon.losses.KLDivergence import KLDivergence
from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.modules.base.DecoderDataParameterizer import DecoderDataParameterizer
from cavachon.modules.base.EncoderLatentParameterizer import EncoderLatentParameterizer
from cavachon.modules.base.HierarchicalEncoder import HierarchicalEncoder
from cavachon.modules.preprocessors import Preprocessor
from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
import warnings

class Component(tf.keras.Model):
  def __init__(
    self,
    inputs: Mapping[Any, tf.keras.Input],
    outputs: Mapping[Any, tf.Tensor],
    modality_names: List[Any],
    distribution_names: Mapping[str, str],
    z_prior_parameterizer: tf.keras.layers.Layer,
    name: str = 'component',
    **kwargs):
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.modality_names = modality_names
    self.distribution_names = distribution_names
    self.z_prior_parameterizer = z_prior_parameterizer

  @classmethod
  def setup_inputs(
      cls,
      modality_names: List[str],
      n_vars: Mapping[str, int],
      z_hat_conditional_dims: Optional[int] = None,
      **kwargs) -> Tuple[Mapping[Any, tf.keras.Input]]:
    inputs = dict()
    hierarchical_encoder_inputs = dict()

    for modality_name in modality_names:
      modality_key = (modality_name, Constants.TENSOR_NAME_X)
      inputs.setdefault(
          modality_key,
          tf.keras.Input(
              shape=(n_vars.get(modality_name), ),
              name=f'{modality_name}_matrix'))
    
    if z_hat_conditional_dims is not None:
      z_hat_conditional = tf.keras.Input(
          shape=(z_hat_conditional_dims, ), 
          name='z_hat_conditional')
      inputs.setdefault(('z_hat_conditional', ), z_hat_conditional)
      hierarchical_encoder_inputs.setdefault('z_hat_conditional', z_hat_conditional)
    
    return inputs, hierarchical_encoder_inputs
  
  @classmethod
  def setup_preprocessor(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str], 
      n_vars: Mapping[str, int],
      n_dims: int = 1024,
      name: str = 'preprocessor',
      **kwargs) -> tf.keras.Model:
    
    return Preprocessor.make(
        modality_names = modality_names,
        distribution_names = distribution_names,
        n_vars = n_vars,
        n_dims = n_dims,
        name = name)

  @classmethod
  def setup_decoders(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str],
      n_vars: Mapping[str, int],
      n_decoder_layers: Union[int, Mapping[str, int]] = 3,
      name: str = 'decoder',
      **kwargs) -> Mapping[Any, tf.keras.Model]: 

    decoders = dict()

    if isinstance(n_decoder_layers, int):
      default_n_decoder_layers = n_decoder_layers
      n_decoder_layers = dict()
    else:
      default_n_decoder_layers = 3

    for modality_name in modality_names:
      decoders.setdefault(
          modality_name,
          DecoderDataParameterizer(
              distribution_name=distribution_names.get(modality_name),
              n_vars=n_vars.get(modality_name),
              n_layers=n_decoder_layers.get(modality_name, default_n_decoder_layers),
              name=f'{name}_{modality_name}'))
    
    return decoders

  @classmethod
  def setup_encoder(
      cls,
      n_latent_dims: int = 5,
      n_encoder_layers: int = 3,
      name: str = 'encoder',
      **kwargs) -> tf.keras.Model:

    return EncoderLatentParameterizer(
        n_layers=n_encoder_layers,
        n_latent_dims=n_latent_dims,
        name=name)
  
  @classmethod
  def setup_hiearchical_encoder(
      cls,
      n_latent_dims: int = 5,
      z_hat_conditional_dims: Optional[int] = None,
      name: str = 'hiearchical_encoder',
      **kwargs) -> tf.keras.Model:
    
    return HierarchicalEncoder.make(
        n_latent_dims=n_latent_dims,
        z_hat_conditional_dims=z_hat_conditional_dims,
        name=name)

  @classmethod
  def setup_z_prior_parameterizer(
      cls,
      n_latent_dims: int = 5,
      n_latent_priors: int = 11,
      name: str = 'z_prior_parameterizer',
      **kwargs) -> tf.keras.Model:

    return MixtureMultivariateNormalDiag(
        event_dims=n_latent_dims,
        n_components=n_latent_priors,
        name=name)

  @classmethod
  def setup_z_sampler(
      cls,
      name: str = 'z_sampler',
      **kwargs) -> tf.keras.Model:

    return MultivariateNormalDiagSampler(name=name)

  @classmethod
  def setup_outputs(
      cls,
      inputs: Mapping[Any, tf.keras.Input],
      hierarchical_encoder_inputs: Mapping[str, tf.keras.Input],
      modality_names: List[str],
      preprocessor: tf.keras.Model,
      encoder: tf.keras.Model,
      hierarchical_encoder: tf.keras.Model,
      z_sampler: Union[tf.keras.Model, tf.keras.layers.Layer],
      decoders: Mapping[str, tf.keras.Model],
      name: str,
      **kwargs) -> Mapping[Any, tf.Tensor]:
      
      outputs = dict()
      preprocessor_outputs = preprocessor(inputs)
      z_parameters = encoder(preprocessor_outputs.get(preprocessor.matrix_key))
      z = z_sampler(z_parameters)
      hierarchical_encoder_inputs.setdefault('z', z)
      z_hat = hierarchical_encoder(hierarchical_encoder_inputs)

      outputs.setdefault((name, 'z'), z)
      outputs.setdefault((name, 'z_hat'), z_hat)
      outputs.setdefault((name, 'z_parameters'), z_parameters)
      
      for modality_name in modality_names:
        decoder_inputs = dict()
        decoder_inputs.setdefault('input', z_hat)
        libsize_key = (modality_name, Constants.TENSOR_NAME_LIBSIZE)
        if libsize_key in preprocessor_outputs:
          decoder_inputs.setdefault('libsize', preprocessor_outputs.get(libsize_key))
        x_parameters = decoders.get(modality_name)(decoder_inputs)
        outputs.setdefault((modality_name, 'x_parameters'), x_parameters)
      
      return outputs

  @classmethod
  def make(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str],
      n_vars: Mapping[str, int],
      n_latent_dims: int = 5,
      n_latent_priors: int = 11,
      n_encoder_layers: int = 3,
      n_decoder_layers: Union[int, Mapping[str, int]] = 3,
      z_hat_conditional_dims: Optional[int] = None,
      name: str = 'component',
      **kwargs) -> tf.keras.Model:

    inputs, hierarchical_encoder_inputs = cls.setup_inputs(
        modality_names = modality_names,
        n_vars = n_vars,
        z_hat_conditional_dims = z_hat_conditional_dims,
        **kwargs)

    encoder = cls.setup_encoder(
        n_latent_dims = n_latent_dims,
        n_encoder_layers = n_encoder_layers,
        name = f'{name}_encoder',
        **kwargs)
    
    preprocessor = cls.setup_preprocessor(
        modality_names = modality_names,
        distribution_names = distribution_names,
        n_vars = n_vars,
        n_dims = encoder.max_n_neurons,
        name = f'{name}_preprocessor',
        **kwargs)

    hierarchical_encoder = cls.setup_hiearchical_encoder(
        n_latent_dims = n_latent_dims,
        z_hat_conditional_dims = z_hat_conditional_dims,
        name = f'{name}_hierarchical_encoder',
        **kwargs)

    z_prior_parameterizer = cls.setup_z_prior_parameterizer(
        n_latent_dims = n_latent_dims,
        n_latent_priors = n_latent_priors,
        name = f'{name}_z_prior_parameterizer',
        **kwargs)

    z_sampler = cls.setup_z_sampler(
        name = f'{name}_z_sampler',
        **kwargs)

    decoders = cls.setup_decoders(
        modality_names = modality_names,
        distribution_names = distribution_names,
        n_vars = n_vars,
        n_decoder_layers = n_decoder_layers,
        name_prefix = name,
        **kwargs)

    outputs = cls.setup_outputs(
        inputs = inputs,
        hierarchical_encoder_inputs = hierarchical_encoder_inputs,
        modality_names = modality_names,
        preprocessor = preprocessor,
        encoder = encoder,
        hierarchical_encoder = hierarchical_encoder,
        z_sampler = z_sampler,
        decoders = decoders,
        name = name,
        **kwargs)

    return cls(
        inputs=inputs,
        outputs=outputs,
        modality_names=modality_names,
        distribution_names=distribution_names,
        z_prior_parameterizer=z_prior_parameterizer,
        name=name,
        **kwargs)

  def compile(
      self,
      optimizer: Union[str, tf.keras.optimizers.Optimizer]='rmsprop',
      **kwargs) -> None:

    if 'loss' not in kwargs:
      loss = OrderedDict()
      kl_divergence_name = 'kl_divergence'
      loss.setdefault(
          kl_divergence_name,
          KLDivergence(name=kl_divergence_name))
      for modality_name in self.modality_names:
        negative_log_data_likelihood_name = f'{modality_name}_negative_log_data_likelihood'
        loss.setdefault(
            negative_log_data_likelihood_name,
            NegativeLogDataLikelihood(
                self.distribution_names.get(modality_name),
                name=negative_log_data_likelihood_name))
      kwargs.setdefault('loss', loss)
    else:
      message = ''.join((
        f'Please make sure the provided custom losses are properly used in train_step() of ' ,
        f'{self.__class__.__name__}.'))
      warnings.warn(message, RuntimeWarning)

    if 'metrics' in kwargs:
      message = ''.join((
        f'Due to the unexpected behaviour with compiled_loss when providing less number ',
        f'of custom losses in Tensorflow 2.8.0. The custom metrics provided to compile() in',
        f'{self.__class__.__name__} will be ignored.'))
      warnings.warn(message, RuntimeWarning)
      kwargs.pop('metrics')

    super().compile(optimizer=optimizer, **kwargs)

  def train_step(self, data):
    with tf.GradientTape() as tape:
      results = self(data, training=True)
      y_true = dict()
      y_pred = dict()
      kl_divergence_name = 'kl_divergence'
      y_true.setdefault(
          kl_divergence_name ,
          self.z_prior_parameterizer(tf.ones((1, 1))))

      y_pred.setdefault(
          kl_divergence_name ,
          tf.concat(
              [results.get((self.name, 'z')), results.get((self.name, 'z_parameters'))],
              axis=-1))

      for modality_name in self.modality_names:
        negative_log_data_likelihood_name = f'{modality_name}_negative_log_data_likelihood'
        modality_key = (modality_name, Constants.TENSOR_NAME_X)
        data = ToDense(modality_key)(data)
        y_true.setdefault(
            negative_log_data_likelihood_name,
            data.get(modality_key))
        y_pred.setdefault(
            negative_log_data_likelihood_name,
            results.get((modality_name, 'x_parameters')))

      loss = self.compiled_loss(y_true, y_pred)
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(y_true, y_pred)
    
    names = ['loss'] + [x.name for x in self.compiled_loss._losses]
    return {name: m.result() for name, m in zip(names, self.metrics)}