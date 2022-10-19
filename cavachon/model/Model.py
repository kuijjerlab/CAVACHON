from cavachon.config.ComponentConfig import ComponentConfig
from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.layers.modifiers import ToDense
from cavachon.losses.KLDivergence import KLDivergence
from cavachon.losses.NegativeLogDataLikelihood import NegativeLogDataLikelihood
from cavachon.modules.components.Component import Component
from cavachon.utils.GeneralUtils import GeneralUtils
from cavachon.utils.TensorUtils import TensorUtils
from tqdm import tqdm
from typing import Any, List, Mapping, Tuple, Union

import numpy as np
import muon as mu
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

  component_configs: List[ComponentConfig]
      the config used to create the components in the model.

  """
  def __init__(
      self,
      inputs: Mapping[Any, tf.keras.Input],
      outputs: Mapping[Any, tf.Tensor],
      components: List[Component],
      component_configs: List[ComponentConfig],
      name: str = 'model',
      **kwargs):
    """Constuctor for Model. Should not be called directly most of the 
    time. Please use make() to create the model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]): 
        inputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, expect to have keys 'z_hat_conditional',
        `modality_name`_matrix, and `modality_name`_libsize (if 
        appplicable).
    
    outputs: Mapping[Any, tf.keras.Input]): 
        outputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, the keys are 
        `component_names`_z, 
        `component_names`_z_hat, 
        `component_names`_z_parameters and 
        `component_names`_`modality_nanes`_x_parameters.
    
    components: List[Component]
      list of components which makes up the model.

    component_configs: List[ComponentConfig]
      the config used to create the components in the model.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'model'.
    
    kwargs: Mapping[str, Any]
        additional parameters for custom models.

    """
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.components: List[Component] = components
    self.component_configs: List[ComponentConfig] = component_configs

  @classmethod
  def setup_inputs(
      cls,
      modality_names: List[str],
      n_vars: Mapping[str, int],
      n_vars_batch_effect: Mapping[str, int],
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
    
    n_vars_batch_effect: Mapping[str, int]
        number of variables for the batch effect tensor. It should 
        be the size of last dimensions of batch effect Tensor. The keys 
        are the modality names, and the values are the corresponding 
        number of variables.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_inputs()

    Returns
    -------
    Mapping[Any, tf.keras.Input]:
        inputs for building tf.keras.Model using Tensorflow functional 
        API, where keys are `modality_name`/matrix, values are the
        tf.keras.Input.

    """
    inputs = dict()
    for modality_name in modality_names:
      modality_matrix_key = f'{modality_name}/{Constants.TENSOR_NAME_X}'
      modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
      inputs.setdefault(
          modality_matrix_key,
          tf.keras.Input(
              shape=(n_vars.get(modality_name), ),
              name=f'{modality_name}/{Constants.TENSOR_NAME_X}'))
      inputs.setdefault(
          modality_batch_key,
          tf.keras.Input(
              shape=(n_vars_batch_effect.get(modality_name), ),
              name=f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'))

    return inputs

  @classmethod
  def setup_components(
      cls,
      component_configs: List[ComponentConfig],
      **kwargs) -> Tuple:
    """Builder function for setting up components. Developers can 
    overwrite this function to create custom Model.

    Parameters
    ----------
    component_configs: List[ComponentConfig]
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
    n_vars_batch_effect = dict()
    for component_config in component_configs:
      modality_names = modality_names.union(
          set(component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES)))
      distributions.update(
          component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES))
      n_vars.update(component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_VARS))
      n_vars_batch_effect.update(component_config.get('n_vars_batch_effect'))
      component_name = component_config.get('name')

      conditioned_on_keys = [
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT
      ]
      conditioned_on_dims_keys = [
        Constants.MODEL_INPUTS_Z_CONDITIONAL_DIMS,
        Constants.MODEL_INPUTS_Z_HAT_CONDITIONAL_DIMS
      ]
      conditioned_on_keys = zip(conditioned_on_keys, conditioned_on_dims_keys)
      for conditioned_on_key, conditioned_on_dims_key in conditioned_on_keys:
        conditioned_on = component_config.get(conditioned_on_key, [])
        if len(conditioned_on) == 0:
          component_config.setdefault(conditioned_on_dims_key, None)
        else:
          conditional_dims = 0
          for conditioned_on_component_name in conditioned_on:
            component = components.get(conditioned_on_component_name)
            conditional_dims += component.z_prior_parameterizer.event_dims
          component_config.setdefault(
              conditioned_on_dims_key,
              conditional_dims)
      components.setdefault(component_name, Component.make(**component_config))
  
    return components, component_configs, modality_names, n_vars, n_vars_batch_effect

  @classmethod
  def setup_outputs(
      cls,
      inputs: Mapping[Any, tf.keras.Input],
      components: List[Component],
      component_configs: List[ComponentConfig],
      **kwargs) -> Mapping[Any, tf.Tensor]:
    """Builder function for setting up outputs. Developers can overwrite 
    this function to create custom Model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]
        inputs created using setup_inputs()
    
    components: List[Component]
      components created by setup_components().

    component_configs: List[ComponentConfig]
      the config used to create the components in the model.

    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_outputs()

    Returns
    -------
    Mapping[Any, tf.Tensor]
        outputs for building tf.keras.Model using Tensorflow functional 
        API.

    """
    z_conditional = dict()
    z_hat_conditional = dict()
    outputs = dict()
    for component_config in component_configs:
      component_inputs = dict()
      component_name = component_config.get('name')
      component = components.get(component_name)
      for modality_name in component.modality_names:
        modality_matrix_key = f"{modality_name}/{Constants.TENSOR_NAME_X}"
        modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
        component_inputs.setdefault(modality_matrix_key, inputs.get(modality_matrix_key))
        component_inputs.setdefault(modality_batch_key, inputs.get(modality_batch_key))

      conditioned_on_keys = [
        Constants.MODULE_INPUTS_CONDITIONED_Z,
        Constants.MODULE_INPUTS_CONDITIONED_Z_HAT
      ]
      conditioned_on_config_keys = [
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z,
        Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT
      ]
      conditioned_on_dict = [
        z_conditional,
        z_hat_conditional
      ]
      conditioned_on_keys = zip(
          conditioned_on_keys, 
          conditioned_on_config_keys, 
          conditioned_on_dict)
      for conditioned_on_key, conditioned_on_config_key, conditioned_on_d in conditioned_on_keys:
        conditional = []
        conditioned_on = component_config.get(conditioned_on_config_key, [])
        if len(conditioned_on) != 0:
          for conditioned_on_component_name in conditioned_on:
            conditional.append(conditioned_on_d.get(conditioned_on_component_name))
          conditional = tf.concat(conditional, axis=-1)
          component_inputs.setdefault(
              conditioned_on_key,
              conditional)

      results = component(component_inputs)
      for key, result in results.items():
        outputs.setdefault(f"{component_name}/{key}", result)
     
      z_conditional.setdefault(
          component_name, 
          results.get(Constants.MODEL_OUTPUTS_Z))

      z_hat_conditional.setdefault(
          component_name,
          results.get(Constants.MODEL_OUTPUTS_Z_HAT))
    
    return outputs

  @classmethod
  def make(
      cls,
      component_configs: List[ComponentConfig],
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
    components, component_configs, modality_names, n_vars, n_vars_batch_effect = cls.setup_components(
        component_configs = component_configs,
        **kwargs)

    inputs = cls.setup_inputs(
        modality_names = modality_names,
        n_vars = n_vars,
        n_vars_batch_effect = n_vars_batch_effect,
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

  def predict(
      self, 
      x: Union[Mapping[str, tf.Tensor], mu.MuData],
      batch_size: int = None,
      **kwargs):
    """Predict based on Mapping[str, tf.Tensor] (with the same format constucted by the dataset
    of DataLoader) or mu.MuData. If provided with mu.MuData, the predicted z and x_parameters will
    be stored in the obsm of each modality.

    Parameters
    ----------
    x: Union[Mapping[str, tf.Tensor], mu.MuData]
        inputs.
    
    batch_size: int, optional
        batch size. If provided with None, will automatically set to 1. Defaults to None.
    
    kwargs: Mapping[str, Any]
        Additional parameters used to compile the model.

    """
    if batch_size is None:
      batch_size = 1
    if issubclass(type(x), mu.MuData):
      outputs = dict()
      use_which_component = dict()
      field_save_x = Constants.CONFIG_FIELD_COMPONENT_MODALITY_SAVE_X
      field_save_z = Constants.CONFIG_FIELD_COMPONENT_MODALITY_SAVE_Z
      save_x = dict()
      save_z = dict()
      for component_config in self.component_configs:
        component_name = component_config.name
        outputs.setdefault(f"{component_name}/z", list())
        modality_names = component_config.get(Constants.CONFIG_FIELD_COMPONENT_N_VARS).keys()
        predict_x = False

        for modality_name in modality_names:
          if component_config.get(field_save_x).get(modality_name):
            predict_x = True

          use_which_component.setdefault(modality_name, [])
          use_which_component.get(modality_name).append(component_name)
          save_x.setdefault(
              f'{component_name}/{modality_name}',
              component_config.get(field_save_x).get(modality_name))
          save_z.setdefault(
              f'{component_name}/{modality_name}',
              component_config.get(field_save_z).get(modality_name))
        if predict_x:
          outputs.setdefault(f"{component_name}/{modality_name}/x_parameters", list())

      dataloader = DataLoader(x, batch_size=batch_size)
      for batch in tqdm(dataloader):
        result = self.predict_on_batch(batch)
        for key in outputs:
          outputs[key].append(result.get(key))
      for key in outputs:
        outputs[key] = np.vstack(outputs[key])
      
      for modality_name, component_names in use_which_component.items():
        for component_name in component_names:
          if save_z.get(f'{component_name}/{modality_name}'):
            x.mod[modality_name].obsm[f'z_{component_name}'] = outputs.get(
                f"{component_name}/z")
          if save_x.get(f'{component_name}/{modality_name}'):
            x.mod[modality_name].obsm[f'x_parameters_{component_name}'] = outputs.get(
                f"{component_name}/{modality_name}/x_parameters")

      return outputs
    else:
      return super.__predict__(x=x, batch_size=batch_size, *args, **kwargs)

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
        kl_divergence_name = f'{component_name}/{Constants.MODEL_LOSS_KL_POSTFIX}'
        loss.setdefault(
            kl_divergence_name,
            KLDivergence(name=kl_divergence_name))

        for modality_name in component_config.get('modality_names'):
          nldl_name = f'{component_name}/{modality_name}/{Constants.MODEL_LOSS_DATA_POSTFIX}'
          distribution_names = component_config.get(
              Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES)
          loss.setdefault(
              nldl_name,
              NegativeLogDataLikelihood(
                  distribution_names.get(modality_name),
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
        kl_divergence_name = f'{component_name}/{Constants.MODEL_LOSS_KL_POSTFIX}'
        component = self.components.get(component_name)

        modality_names = component_config.get(Constants.CONFIG_FIELD_COMPONENT_MODALITY_NAMES)
        y_true.setdefault(
            kl_divergence_name,
            component.z_prior_parameterizer(tf.ones((1, 1))))
        
        z_key = f'{component_name}/{Constants.MODEL_OUTPUTS_Z}'
        z_params_key = f'{component_name}/{Constants.MODEL_OUTPUTS_Z_PARAMS}'
            
        y_pred.setdefault(
            kl_divergence_name,
            tf.concat([results.get(z_key), results.get(z_params_key)], axis=-1))
        for modality_name in modality_names:
          nldl_name = f'{component_name}/{modality_name}/{Constants.MODEL_LOSS_DATA_POSTFIX}'
          modality_key = f"{modality_name}/{Constants.TENSOR_NAME_X}"
          data = ToDense(modality_key)(data)
          y_true.setdefault(
              nldl_name,
              data.get(modality_key))
          y_pred.setdefault(
              nldl_name,
              results.get(f"{component_name}/{modality_name}/{Constants.MODEL_OUTPUTS_X_PARAMS}"))
      
      loss = self.compiled_loss(y_true, y_pred)
      gradients = tape.gradient(loss, self.trainable_variables)
      gradients = TensorUtils.remove_nan_gradients(gradients)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(y_true, y_pred)
    
    names = ['loss'] + [x.name for x in self.compiled_loss._losses]
    return {name: m.result() for name, m in zip(names, self.metrics)}
  
