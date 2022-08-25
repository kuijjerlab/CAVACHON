from cavachon.environment.Constants import Constants
from cavachon.losses.KLDivergence import KLDivergence
from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.modules.conditionals.ConditionalModuleFunctionalBackup import ConditionalModule
from cavachon.utils.TensorUtils import TensorUtils
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Any, Collection, Dict, List, Mapping, MutableMapping, Optional, Union

import tensorflow as tf
import warnings

class Model(tf.keras.Model):
  def __init__(
      self,
      module_names: List[str] = [],
      module_types: Dict[str, str] = dict(),
      modality_names: Dict[str, Collection[str]] = dict(),
      distribution_names: Dict[str, List[Union[str, str]]] = dict(), 
      n_vars: Dict[str, Mapping[str, int]] = dict(),
      n_latent_dims: Dict[str, Mapping[str, int]] = dict(),     
      n_priors: Dict[str, Mapping[str, int]] = dict(),          
      n_encoder_layers: Dict[str, Mapping[str, int]] = dict(), 
      n_decoder_layers: Dict[str, Mapping[str, int]] = dict(),  
      name: str = 'Model'):
    super().__init__(name=name)
    self.preprocessors: List[tf.keras.Model] = list()
    self.conditional_modules: List[ConditionalModule] = list()
    
    distribution_of_modality = dict()
    conditional_module_name = None
    for order, module_name in enumerate(module_names):
      module_type = module_types.get(module_name)
      module_class = ReflectionHandler.get_class_by_name(module_type, 'modules/conditionals')
      module_modality_names = modality_names.get(module_name, list())
      module_distribution_names = distribution_names.get(module_name, dict())
      module_n_vars = n_vars.get(module_name, dict())
      module_n_latent_dims = n_latent_dims.get(module_name, dict())
      module_n_priors = n_priors.get(module_name, dict())
      module_n_encoder_layers = n_encoder_layers.get(module_name, dict())
      module_n_decoder_layers = n_decoder_layers.get(module_name, dict())
      
      self.conditional_modules.append(
        module_class(
          order=order,
          modality_names=module_modality_names,
          distribution_names=module_distribution_names,
          n_vars=module_n_vars,
          n_latent_dims=module_n_latent_dims,
          n_priors=module_n_priors, 
          n_encoder_layers=module_n_encoder_layers,
          n_decoder_layers=module_n_decoder_layers, 
          conditional_module_name=conditional_module_name,
          name=module_name
        )
      )
      conditional_module_name = module_name
            
      for modality_name in module_modality_names:
        distribution_of_modality.setdefault(
            modality_name,
            module_distribution_names.get(modality_name))
    
    for modality_name, distribution_name in distribution_of_modality.items():
      preprocessor = ReflectionHandler.get_class_by_name(distribution_name, 'modules/preprocessors')
      self.preprocessors.append(
        preprocessor((modality_name, Constants.TENSOR_NAME_X))
      )

  def compile(
      self,
      optimizer: Union[str, tf.keras.optimizers.Optimizer]='rmsprop',
      **kwargs) -> None:
    if 'loss' in kwargs:
      message = ''.join((
          f"Note that the {kwargs.get('loss')} provided to loss will not be compiled to ",
          f"LossContainer in {self.__class__.__name__}. Instead, the call() function implemented ",
          f"in the {kwargs.get('loss')} will be used directly when computing the loss function."
      ))
      warnings.warn(message, RuntimeWarning)
      kl_divergence, negative_log_data_likelihood = kwargs.pop('loss')
      for module in self.conditional_modules:
        module.setup_elbo_estimator(kl_divergence, negative_log_data_likelihood)
    
    if 'metrics' in kwargs:
      message = ''.join((
          f"Note that the {kwargs.get('metrics')} provided to loss will not be used during ",
          f"training process. Instead, the result of loss will be shown."
      ))
      warnings.warn(message, RuntimeWarning)
      kwargs.pop('metrics')

    super().compile(optimizer=optimizer, **kwargs)

  def call(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:

    for module in self.conditional_modules:
      inputs.update(module(inputs, training=training, mask=mask))

    return inputs

  def preprocess(
      self,
      inputs: MutableMapping[Any, tf.Tensor],
      training: bool = False,
      mask: Optional[tf.Tensor] = None) -> MutableMapping[Any, tf.Tensor]:
    for preprocessor in self.preprocessors:
      inputs = preprocessor(inputs, training, mask)
    
    return inputs

  def train_step(self, data) -> Dict[str, float]:
    with tf.GradientTape() as tape:
      total_elbo = 0
      metrics = dict()
      preprocessed_data = self.preprocess(data, training=True)
      for module in self.conditional_modules:
        elbo, results = module.compute_elbo(preprocessed_data, training=True)
        elbo += total_elbo
        metrics.update(results)

      gradients = tape.gradient(-elbo, self.trainable_variables)
      gradients = TensorUtils.remove_nan_gradients(gradients)
    
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return metrics

  @staticmethod
  def prepare_kl_divergence_input(z: tf.Tensor, z_parameters: tf.Tensor) -> tf.Tensor:
    return tf.concat([z, z_parameters], axis=-1)