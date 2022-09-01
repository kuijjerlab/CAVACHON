from cavachon.environment.Constants import Constants
from cavachon.layers.modifiers import ToDense
from cavachon.losses.KLDivergence import KLDivergence
from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.modules.components.Component import Component
from cavachon.utils.GeneralUtils import GeneralUtils
from cavachon.utils.TensorUtils import TensorUtils
from typing import Any, Dict, List, Mapping, Iterable, Tuple, Union

import tensorflow as tf
import warnings

class Model(tf.keras.Model):
  def __init__(
      self,
      inputs: Mapping[Any, tf.keras.Input],
      outputs: Mapping[Any, tf.Tensor],
      components: List[Component],
      component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]],
      name: str = 'model',
      **kwargs):
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.components = components
    self.component_configs = component_configs

  @classmethod
  def setup_inputs(
      cls,
      modality_names: List[str],
      n_vars: Mapping[str, int]) -> Mapping[Any, tf.keras.Input]:

    inputs = dict()
    for modality_name in modality_names:
      modality_key = (modality_name, Constants.TENSOR_NAME_X)
      inputs.setdefault(
          modality_key,
          tf.keras.Input(
              shape=(n_vars.get(modality_name), ),
              name=f'{modality_name}_matrix'))

    return inputs

  @classmethod
  def setup_components(
      cls,
      component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]],
      **kwargs) -> Tuple:

    component_configs = GeneralUtils.order_components(component_configs)
    components = dict()
    modality_names = set()
    distributions = dict()
    n_vars = dict()
    for component_config in component_configs:
      modality_names = modality_names.union(set(component_config.get('modality_names')))
      distributions.update(component_config.get('distribution_names'))
      n_vars.update(component_config.get('n_vars'))
      component_name = component_config.get('name')
      conditioned_on = component_config.get('conditioned_on', [])
      if len(conditioned_on) == 0:
        component_config.setdefault('z_hat_conditional_dims', None)
        components.setdefault(component_name, Component.make(**component_config))
      else:
        z_hat_conditional_dims = 0
        for conditioned_on_component_name in conditioned_on:
          component = components.get(conditioned_on_component_name)
          z_hat_conditional_dims += component.z_prior_parameterizer.event_dims
        component_config.setdefault('z_hat_conditional_dims', z_hat_conditional_dims)
        components.setdefault(component_name, Component.make(**component_config))
    
    return components, component_configs, modality_names, n_vars

  @classmethod
  def setup_outputs(
      cls,
      inputs: Mapping[Any, tf.keras.Input],
      components: List[Component],
      component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]],
      **kwargs) -> Mapping[Any, tf.Tensor]:
    
    z_hat_conditional = dict()
    outputs = dict()
    for component_config in component_configs:
      component_inputs = dict()
      component_name = component_config.get('name')
      conditioned_on = component_config.get('conditioned_on', [])
      component = components.get(component_name)
      for modality_name in component.modality_names:
        modality_key = (modality_name, Constants.TENSOR_NAME_X)
        component_inputs.setdefault(modality_key, inputs.get(modality_key))
      
      if len(conditioned_on) != 0:
        z_hat = []
        for conditioned_on_component_name in conditioned_on:
          z_hat.append(z_hat_conditional.get(conditioned_on_component_name))
        z_hat = tf.concat(z_hat, axis=-1)
        component_inputs.setdefault(('z_hat_conditional', ), z_hat)
      
      results = component(component_inputs)
      for key, result in results.items():
        outputs.setdefault((component_name, ) + key, result)
      z_hat_conditional.setdefault(component_name, results.get((component_name, 'z_hat')))
    
    return outputs

  @classmethod
  def make(
      cls,
      component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]],
      name: str = 'cavachon',
      **kwargs) -> tf.keras.Model:

    components, component_configs, modality_names, n_vars = cls.setup_components(
        component_configs = component_configs,
        **kwargs)

    inputs = cls.setup_inputs(
        modality_names = modality_names,
        n_vars = n_vars,
        **kwargs)

    outputs = cls.setup_outputs(
        inputs = inputs,
        components = components,
        component_configs = component_configs,
        **kwargs)

    return cls(
        inputs=inputs,
        outputs=outputs,
        name=name,
        components=components,
        component_configs=component_configs)

  def compile(
      self,
      optimizer: Union[str, tf.keras.optimizers.Optimizer]='rmsprop',
      **kwargs) -> None:

    if 'loss' not in kwargs:
      loss = dict()
      for component_config in self.component_configs:
        component_name = component_config.get('name')
        kl_divergence_name = f'{component_name}_kl_divergence'
        loss.setdefault(
            kl_divergence_name,
            KLDivergence(name=kl_divergence_name))

        for modality_name in component_config.get('modality_names'):
          nldl_name = f'{component_name}_{modality_name}_negative_log_data_likelihood'
          loss.setdefault(
              nldl_name,
              NegativeLogDataLikelihood(
                  component_config.get('distribution_names').get(modality_name),
                  name=nldl_name))
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
  
      for component_config in self.component_configs:
        component_name = component_config.get('name')
        kl_divergence_name = f'{component_name}_kl_divergence'
        component = self.components.get(component_name)
        modality_names = component_config.get('modality_names')
        y_true.setdefault(
            kl_divergence_name,
            component.z_prior_parameterizer(tf.ones((1, 1))))
        y_pred.setdefault(
            kl_divergence_name,
            tf.concat(
                [results.get((component_name, component_name, 'z')), results.get((component_name, component_name, 'z_parameters'))],
                axis=-1))
        for modality_name in modality_names:
          nldl_name = f'{component_name}_{modality_name}_negative_log_data_likelihood'
          modality_key = (modality_name, Constants.TENSOR_NAME_X)
          data = ToDense(modality_key)(data)
          y_true.setdefault(
              nldl_name,
              data.get(modality_key))
          y_pred.setdefault(
              nldl_name,
              results.get((component_name, modality_name, 'x_parameters')))
      
      loss = self.compiled_loss(y_true, y_pred)
      gradients = tape.gradient(loss, self.trainable_variables)
      gradients = TensorUtils.remove_nan_gradients(gradients)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(y_true, y_pred)
    
    names = ['loss'] + [x.name for x in self.compiled_loss._losses]
    return {name: m.result() for name, m in zip(names, self.metrics)}