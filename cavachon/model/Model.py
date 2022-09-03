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
  """Model
  
  Main CAVACHON model. It consists of multiple Components and the
  dependency between them.

  Attibutes
  ---------
  components: List[Component]
      list of components which makes up the model.

  component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
      the config used to create the components in the model.

  """
  def __init__(
      self,
      inputs: Mapping[Any, tf.keras.Input],
      outputs: Mapping[Any, tf.Tensor],
      components: List[Component],
      component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]],
      name: str = 'model',
      **kwargs):
    """Constuctro for Model. Should not be called directly most of the 
    time. Please use make() to create the model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]): 
        inputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, expect to have keys ('z_hat_conditional', ),
        (modality_name, Constants.TENSOR_NAME_X), and
        (modality_name, Constants.LIBSIZE) (if appplicable).
    
    outputs: Mapping[Any, tf.keras.Input]): 
        outputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, the keys are 
        (component_names, component_names, 'z'), 
        (component_names, component_names, 'z_hat'), 
        (component_names, component_names, 'z_parameters') 
        and (component_names, modality_nanes, 'x_parameters').
    
    components: List[Component]
      list of components which makes up the model.

    component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
      the config used to create the components in the model.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'model'.
    
    kwargs: Mapping[str, Any]
        additional parameters for custom models.

    """
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.components: List[Component] = components
    self.component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]] = component_configs

  @classmethod
  def setup_inputs(
      cls,
      modality_names: List[str],
      n_vars: Mapping[str, int],
      **kwargs) -> Mapping[Any, tf.keras.Input]:
    """Builder function for setting up inputs. Developers can overwrite 
    this function to create custom Model.

    Parameters
    ----------
    modality_names: str
        names of the modalities used in the model. 
    
    n_vars: Mapping[str, int]
        number of variables for the inputs data distribution. It should 
        be the size of last dimensions of inputs Tensor. The keys are 
        the modality names, and the values are the corresponding number
        of variables.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_inputs()

    Returns
    -------
    Mapping[Any, tf.keras.Input]:
        inputs for building tf.keras.Model using Tensorflow functional 
        API.

    """
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
    """Builder function for setting up components. Developers can 
    overwrite this function to create custom Model.

    Parameters
    ----------
    component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
        the config used to create the components in the model.

    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_components()

    Returns
    -------
    Tuple
        the first element is the list of created components, the second
        element is the component configs but reordered based on the
        number of breadth first search successors in the dependency
        direct acyclic graph. The third element is the list of names of
        all modalities used in the model. The last element is the 
        Mapping of number of variables for each modality, where the 
        keys are the modality names.

    """
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
    """Builder function for setting up outputs. Developers can overwrite 
    this function to create custom Model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]
        inputs created using setup_inputs()
    
    components: List[Component]
      components created by setup_components().

    component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
      the config used to create the components in the model.

    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_outputs()

    Returns
    -------
    Mapping[Any, tf.Tensor]
        outputs for building tf.keras.Model using Tensorflow functional 
        API.

    """

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
    """Make the tf.keras.Model using the functional API of Tensorflow.

    Parameters
    ----------
    component_configs: Union[Iterable[Dict[str, Any]], Dict[str, Dict[str, Any]]]
      the config used to create the components in the model.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'component'.

    kwargs: Mapping[str, Any]
        additional parameters used for the builder functions.

    Returns
    -------
    tf.keras.Model
        created model using Tensorflow functional API.

    """
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
      **kwargs) -> None:
    """Compile the model before training. Note that the 'metrics' will 
    be ignored in Model due to some unexpected behaviour for Tensorflow. 
    The 'loss' will be setup automatically if not provided.

    Parameters
    ----------
    kwargs: Mapping[str, Any]
        Additional parameters used to compile the model.

    """
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

    super().compile(**kwargs)

  def train_step(self, data: Mapping[Any, tf.Tensor]) -> Mapping[str, float]:
    """Training step for one iteration. The trainable variables in the
    Model will be trained once after calling this function.

    Parameters
    ----------
    data: Mapping[Any, tf.Tensor]
        input data with stucture specified with self.inputs.

    Returns
    -------
    Mapping[str, float]
        losses trained in the training iteration, where the keys are the
        names of the losses.
        
    """
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