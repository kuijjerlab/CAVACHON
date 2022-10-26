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
from cavachon.utils.TensorUtils import TensorUtils
from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Tuple, Union

import tensorflow as tf
import warnings

class Component(tf.keras.Model):
  """Component

  Main component used in CAVACHON model. By defaults, it includes a
  Preprocessor, an EncoderLatentParameterizer, a HierarchicalEncoder, a
  Sampler, multiple DecoderDataParameterizer (for each modality) and a 
  Parameterizer of the priors for latent distributions. Among the 
  modules, the Preprocessor and HierarchicalEncoder are implemented 
  using the Tensorflow functional API.

  Attributes
  ----------
  modality_names: str
      names of the modalities used in the component. 

  distribution_names: Mapping[str, str]
      names of the distributions for each modality. The keys are the
      names of the modalities, and the values are the corresponding 
      distribution names. This is used to automatically find the 
      parameterizer in modules.parameterizers for data distributions.
  
  encoder: tf.keras.Model
      encoder neural network.

  z_prior_parameterizer: tf.keras.layers.Layer
      parameterizer used for the priors in latent distributions. Used 
      when computing the KLDivergence.

  hierarchical_encoder: tf.karas.Model
      hierarchical encoder used to encode z_hat hierarchically through 
      the dependency between components.
  
  decoders: Mapping[str, tf.keras.Model]
      decoder neural networks. The keys are the name of the modality,
      the values are the corresponding decoder neural network.
      
  name: str, optional
      Name for the tensorflow model. Defaults to 'component'.
  """
  def __init__(
    self,
    inputs: Mapping[Any, tf.keras.Input],
    outputs: Mapping[Any, tf.Tensor],
    modality_names: List[Any],
    distribution_names: Mapping[str, str],
    encoder: tf.keras.Model,
    z_prior_parameterizer: tf.keras.layers.Layer,
    hierarchical_encoder: tf.keras.Model,
    decoders: Mapping[str, tf.keras.Model],
    name: str = 'component',
    **kwargs):
    """Constructor for Component. Should not be called directly most of 
    the time. Please use make() to create the model.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]): 
        inputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, expect to have keys `modality_name`/matrix,
        'z_conditional' (if applicable), 'z_hat_conditional' (if 
        applicable) and `modality_name`/libsize (if appplicable).
    
    outputs: Mapping[Any, tf.Tensor]
        outputs for building tf.keras.Model using Tensorflow functional 
        API. By defaults, the keys are `model_name`/z, 
        `model_name`/z_hat, `model_name`/z_parameters and 
        `modality_names`/x_parameters.

    modality_names: str
      names of the modalities used in the component. 

    distribution_names: str
        names of the distributions for each modality, this is used to 
        automatically find the parameterizer in modules.parameterizers
        for data distributions.
  
    encoder: tf.keras.Model
        encoder neural network.

    z_prior_parameterizer: tf.keras.layers.Layer
        parameterizer used for the priors in latent distributions. Used 
        when computing the KLDivergence.

    hierarchical_encoder: tf.karas.Model
        hierarchical encoder used to encode z_hat hierarchically through 
        the dependency between components.
    
    decoders: Mapping[str, tf.keras.Model]
        decoder neural networks. The keys are the name of the modality,
        the values are the corresponding decoder neural network.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'component'.

    kwargs: Mapping[str, Any]
        additional parameters for custom components.

    """
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.modality_names = modality_names
    self.distribution_names = distribution_names
    self.encoder = encoder
    self.z_prior_parameterizer = z_prior_parameterizer
    self.hierarchical_encoder = hierarchical_encoder
    self.decoders = decoders

  @classmethod
  def setup_inputs(
      cls,
      modality_names: List[str],
      n_vars: Mapping[str, int],
      n_vars_batch_effect: Mapping[str, int],
      z_conditional_dims: Optional[int] = None,
      z_hat_conditional_dims: Optional[int] = None,
      **kwargs) -> Tuple[Mapping[Any, tf.keras.Input]]:
    """Builder function for setting up inputs. Developers can overwrite 
    this function to create custom Component.

    Parameters
    ----------
    modality_names: str
        names of the modalities used in the component. 
    
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
        
    z_conditional_dims: int, optional
        dimension of z from the components of the dependency. None if 
        the component does not depends on any other components. 
        Defaults to None.

    z_hat_conditional_dims: int, optional
        dimension of z_hat from the components of the dependency. None 
        if the component does not depends on any other components. 
        Defaults to None.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_inputs()

    Returns
    -------
    Tuple[Mapping[Any, tf.keras.Input]]:
        the first element in the tuple is the inputs (for the entire
        component) and the second element is the inputs for the 
        HierarchicalEncoder.

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
    
    if z_conditional_dims is not None:
      z_conditional = tf.keras.Input(
          shape=(z_conditional_dims, ), 
          name=Constants.MODULE_INPUTS_CONDITIONED_Z)
      inputs.setdefault(Constants.MODULE_INPUTS_CONDITIONED_Z, z_conditional)

    if z_hat_conditional_dims is not None:
      z_hat_conditional = tf.keras.Input(
          shape=(z_hat_conditional_dims, ), 
          name=Constants.MODULE_INPUTS_CONDITIONED_Z_HAT)
      inputs.setdefault(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, z_hat_conditional)
    
    return inputs
  
  @classmethod
  def setup_preprocessor(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str], 
      n_vars: Mapping[str, int],
      n_dims: int = 1024,
      name: str = 'preprocessor',
      **kwargs) -> tf.keras.Model:
    """Builder function for setting up preprocessor. Developers can 
    overwrite this function to create custom Component.

    Parameters
    ----------
    modality_names: str
        names of the modalities used in the component. 
  
    distribution_names: Mapping[str, str]
        names of the distributions for each modality. The keys are the
        names of the modalities, and the values are the corresponding 
        distribution names.
         
    n_vars: Mapping[str, int]
        number of variables for the inputs data distribution. It should 
        be the size of last dimensions of inputs Tensor. The keys are 
        the modality names, and the values are the corresponding number
        of variables.
    
    n_dims: int
        number of dimensions to reduced before concatenating the inputs.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'preprocessor'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_preprocessors()

    Returns
    -------
    tf.keras.Model
        created preprocessor.

    """
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
    """Builder function for setting up decoders. Developers can 
    overwrite this function to create custom Component.

    Parameters
    ----------
    modality_names: str
        names of the modalities used in the component. 
  
    distribution_names: Mapping[str, str]
        names of the distributions for each modality. The keys are the
        names of the modalities, and the values are the corresponding 
        distribution names.
         
    n_vars: Mapping[str, int]
        number of variables for the inputs data distribution. It should 
        be the size of last dimensions of inputs Tensor. The keys are 
        the modality names, and the values are the corresponding number
        of variables.
    
    n_decoder_layers: Union[int, Mapping[str, int]]
        number of decoder layers for each modality. If provided with a
        Mapping, the keys are the modality names, and the values are the
        corresponding number of variables. Defaults to 3.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'preprocessor'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_decoders()

    Returns
    -------
    Mapping[Any, tf.keras.Model]:
        the keys for the Mapping is the names of the modalities, and 
        the values are the corresponding decoders for the modalities.

    """

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
              name=f'{name}/{modality_name}'))
    
    return decoders

  @classmethod
  def setup_encoder(
      cls,
      n_latent_dims: int = 5,
      n_encoder_layers: int = 3,
      name: str = 'encoder',
      **kwargs) -> tf.keras.Model:
    """Builder function for setting up encoder. Developers can overwrite 
    this function to create custom Component.

    Parameters
    ----------   
    n_latent_dims: int
        number of latent dimensions. Defaults to 5.
    
    n_encoder_layers: int
        number of encoder layers. Defaults to 3.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'encoder'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_encoders()

    Returns
    -------
    tf.keras.Model:
        created encoder.

    """

    return EncoderLatentParameterizer(
        n_layers=n_encoder_layers,
        n_latent_dims=n_latent_dims,
        name=name)
  
  @classmethod
  def setup_hierarchical_encoder(
      cls,
      n_latent_dims: int = 5,
      is_conditioned_on_z: bool = False,
      is_conditioned_on_z_hat: bool = False,
      progressive_iterations: int = 5000,
      name: str = 'hiearchical_encoder',
      **kwargs) -> tf.keras.Model:
    """Builder function for setting up hierarchical encoder. Developers 
    can overwrite this function to create custom Component.

    Parameters
    ----------   
    n_latent_dims: int
        number of latent dimensions. Defaults to 5.

    conditioned_on_z: bool, optional
        use latent representation from the conditioned components.
        Defaults to False.

    conditioned_on_z_hat: bool, optional
        use transformed latent representation (contains information of 
        all ancestor of conditioned components) from the conditioned 
        components. Defaults to False.

    prorgressive_iterations: int, optional
        total iterations for progressive training. Defaults to 5000.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'encoder'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_encoders()

    Returns
    -------
    tf.keras.Model:
        created hierarchical encoder.

    """
    return HierarchicalEncoder(
        n_latent_dims=n_latent_dims,
        is_conditioned_on_z=is_conditioned_on_z,
        is_conditioned_on_z_hat=is_conditioned_on_z_hat,
        progressive_step=progressive_iterations,
        name=name)

  @classmethod
  def setup_z_prior_parameterizer(
      cls,
      n_latent_dims: int = 5,
      n_latent_priors: int = 11,
      name: str = 'z_prior_parameterizer',
      **kwargs) -> tf.keras.Model:
    """Builder function for setting up the parameterizer of the priors
    for latent distributions. Developers can overwrite this function to 
    create custom Component.

    Parameters
    ----------   
    n_latent_dims: int
        number of latent dimensions. Defaults to 5.
    
    n_latent_priors: int
        number of priors for the latent distributions. To make the model
        identifiable, set this number to at least 2 * n_latent_dims + 1.
        Defaults to 11.

    name: str, optional:
        Name for the tensorflow model. Defaults to 
        'z_prior_parameterizer'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom 
        setup_z_prior_parameterizer()

    Returns
    -------
    tf.keras.Model:
        created parameterizer of the priors for latent distributions.

    """
    return MixtureMultivariateNormalDiag(
        event_dims=n_latent_dims,
        n_components=n_latent_priors,
        name=name)

  @classmethod
  def setup_z_sampler(
      cls,
      name: str = 'z_sampler',
      **kwargs) -> tf.keras.layers.Layer:
    """Builder function for setting up the sampler for the latent 
    distributions. Developers can overwrite this function to create 
    custom Component.

    Parameters
    ----------   
    name: str, optional:
        Name for the tensorflow model. Defaults to 
        'z_sampler'.
    
    kwargs: Mapping[str, Any]
        additional parameters used for custom 
        setup_z_sampler()

    Returns
    -------
    tf.keras.Model:
        created sampler for the latent distributions.

    """
    return MultivariateNormalDiagSampler(name=name)

  @classmethod
  def setup_outputs(
      cls,
      inputs: Mapping[Any, tf.keras.Input],
      modality_names: List[str],
      preprocessor: tf.keras.Model,
      encoder: tf.keras.Model,
      hierarchical_encoder: tf.keras.Model,
      z_sampler: Union[tf.keras.Model, tf.keras.layers.Layer],
      decoders: Mapping[str, tf.keras.Model],
      **kwargs) -> Mapping[Any, tf.Tensor]:
    """Builder function for setting up outputs. Developers can overwrite 
    this function to create custom Component.

    Parameters
    ----------
    inputs: Mapping[Any, tf.keras.Input]
        inputs created using setup_inputs()
      
    hierarchical_encoder_inputs: Mapping[str, tf.keras.Input]
        inputs for hierarchical encoder created using setup_inputs()

    modality_names: str
        names of the modalities used in the component. 
        
    preprocessor: tf.keras.Model
        preprocessor created using setup_preprocessor()

    encoder (tf.keras.Model)
        encoder created using setup_encoder()
    
    hierarchical_encoder (tf.keras.Model)
        hierarhical encoder created using setup_hierarchical_encoder()

    z_sampler: Union[tf.keras.Model, tf.keras.layers.Layer]
        sampler for latent distributions using setup_z_sampler()

    decoders: Mapping[str, tf.keras.Model]
        decoders created using setup_decoders()

    kwargs: Mapping[str, Any]
        additional parameters used for custom setup_outputs()

    Returns
    -------
    Mapping[Any, tf.Tensor]
        outputs for building tf.keras.Model using Tensorflow functional 
        API.

    """
    outputs = dict()
    hierarchical_encoder_inputs = dict()
    preprocessor_inputs = dict()
    preprocessor_inputs.update(inputs)
    preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z, None)
    preprocessor_inputs.pop(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None)

    for modality_name in modality_names:
      modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
      preprocessor_inputs.pop(modality_batch_key, None)

    preprocessor_outputs = preprocessor(preprocessor_inputs)
    z_parameters = encoder(preprocessor_outputs.get(preprocessor.matrix_key))
    z = z_sampler(z_parameters)

    hierarchical_encoder_inputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
    if Constants.MODULE_INPUTS_CONDITIONED_Z in inputs:
        hierarchical_encoder_inputs.setdefault(
            Constants.MODULE_INPUTS_CONDITIONED_Z,
            inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z))
    if Constants.MODULE_INPUTS_CONDITIONED_Z_HAT in inputs:
        hierarchical_encoder_inputs.setdefault(
            Constants.MODULE_INPUTS_CONDITIONED_Z_HAT,
            inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT))
    z_hat = hierarchical_encoder(hierarchical_encoder_inputs)

    outputs.setdefault(Constants.MODEL_OUTPUTS_Z, z)
    outputs.setdefault(Constants.MODEL_OUTPUTS_Z_HAT, z_hat)
    outputs.setdefault(Constants.MODEL_OUTPUTS_Z_PARAMS, z_parameters)
    
    for modality_name in modality_names:
      modality_batch_key = f'{modality_name}/{Constants.TENSOR_NAME_BATCH}'
      decoder_inputs = dict()
      decoder_inputs.setdefault(
          Constants.TENSOR_NAME_X,
          tf.concat([z_hat, inputs.get(modality_batch_key)], axis=-1))
      libsize_key = f'{modality_name}/{Constants.TENSOR_NAME_X}/{Constants.TENSOR_NAME_LIBSIZE}'
      if libsize_key in preprocessor_outputs:
        decoder_inputs.setdefault(Constants.TENSOR_NAME_LIBSIZE, preprocessor_outputs.get(libsize_key))
      x_parameters = decoders.get(modality_name)(decoder_inputs)
      outputs.setdefault(f"{modality_name}/{Constants.MODEL_OUTPUTS_X_PARAMS}", x_parameters)
    
    return outputs

  @classmethod
  def make(
      cls,
      modality_names: List[str],
      distribution_names: Mapping[str, str],
      n_vars: Mapping[str, int],
      n_vars_batch_effect: Mapping[str, int],
      n_latent_dims: int = 5,
      n_latent_priors: int = 11,
      n_encoder_layers: int = 3,
      n_decoder_layers: Union[int, Mapping[str, int]] = 3,
      z_conditional_dims: Optional[int] = None,
      z_hat_conditional_dims: Optional[int] = None,
      progressive_iterations: int = 5000,
      name: str = 'component',
      **kwargs) -> tf.keras.Model:
    """Make the tf.keras.Model using the functional API of Tensorflow.

    Parameters
    ----------
    modality_names: str
        names of the modalities used in the component. 
  
    distribution_names: Mapping[str, str]
        names of the distributions for each modality. The keys are the
        names of the modalities, and the values are the corresponding 
        distribution names.
    
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

    n_latent_dims: int, optional
        number of latent dimensions. Defaults to 5.
    
    n_latent_priors: int
        number of priors for the latent distributions. To make the model
        identifiable, set this number to at least 2 * n_latent_dims + 1.
        Defaults to 11.

    n_encoder_layers: int
        number of encoder layers. Defaults to 3.

    n_decoder_layers: Union[int, Mapping[str, int]]
        number of decoder layers for each modality. If provided with a
        Mapping, the keys are the modality names, and the values are the
        corresponding number of decoder layers. Defaults to 3.
    
    z_conditional_dims: int, optional
        dimension of z from the components of the dependency. None 
        if the component does not depends on any other components. 
        Defaults to None.

    z_hat_conditional_dims: int, optional
        dimension of z_hat from the components of the dependency. None 
        if the component does not depends on any other components. 
        Defaults to None.

    prorgressive_iterations: int, optional
        total iterations for progressive training. Defaults to 5000.

    name: str, optional:
        Name for the tensorflow model. Defaults to 'component'.

    kwargs: Mapping[str, Any]
        additional parameters used for the builder functions.

    Returns
    -------
    tf.keras.Model
        created model using Tensorflow functional API.

    """
    inputs = cls.setup_inputs(
        modality_names = modality_names,
        n_vars = n_vars,
        n_vars_batch_effect = n_vars_batch_effect,
        z_conditional_dims=z_conditional_dims,
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

    hierarchical_encoder = cls.setup_hierarchical_encoder(
        n_latent_dims = n_latent_dims,
        is_conditioned_on_z = z_conditional_dims is not None,
        is_conditioned_on_z_hat = z_hat_conditional_dims is not None,
        progressive_iterations = progressive_iterations,
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
        modality_names = modality_names,
        preprocessor = preprocessor,
        encoder = encoder,
        hierarchical_encoder = hierarchical_encoder,
        z_sampler = z_sampler,
        decoders = decoders,
        **kwargs)

    return cls(
        inputs=inputs,
        outputs=outputs,
        modality_names=modality_names,
        distribution_names=distribution_names,
        encoder=encoder,
        z_prior_parameterizer=z_prior_parameterizer,
        hierarchical_encoder=hierarchical_encoder,
        decoders=decoders,
        name=name,
        **kwargs)

  def compile(
      self,
      **kwargs) -> None:
    """Compile the model before training. Note that the 'metrics' will 
    be ignored in Model becaus of the incompatibility with Tensorflow
    API. The 'loss' will be setup automatically if not provided.

    Parameters
    ----------
    kwargs: Mapping[str, Any]
        Additional parameters used to compile the model.

    """
    loss_weights = kwargs.get('loss_weights', dict())
    kwargs.pop('loss_weights', None)
    
    if 'loss' not in kwargs:
      loss = OrderedDict()
      kl_divergence_name = Constants.MODEL_LOSS_KL_POSTFIX
      loss.setdefault(
          kl_divergence_name,
          KLDivergence(loss_weights.get(kl_divergence_name, 1.0), name=kl_divergence_name))
      for modality_name in self.modality_names:
        nldl_name = f'{modality_name}/{Constants.MODEL_LOSS_DATA_POSTFIX}'
        loss.setdefault(
            nldl_name,
            NegativeLogDataLikelihood(
                self.distribution_names.get(modality_name),
                loss_weights.get(nldl_name, 1.0),
                name=nldl_name))
      kwargs.setdefault('loss', loss)
    else:
      message = ''.join((
        f'Please make sure the provided custom losses are properly used in train_step() of ' ,
        f'{self.__class__.__name__}.'))
      warnings.warn(message, RuntimeWarning)

    if 'metrics' in kwargs:
      message = ''.join((
        f'Due to the incompatibility of the compiled_loss with Tensorflow 2.8.1 (as the model ',
        f'requires outputs from multiple components to compute the KLDivergence), The custom ',
        f'metrics provided to compile() in {self.__class__.__name__} will be ignored.'))
      warnings.warn(message, RuntimeWarning)
      kwargs.pop('metrics')

    super().compile(**kwargs)

  def train_step(self, data: Mapping[Any, tf.Tensor]) -> Mapping[str, float]:
    """Training step for one iteration. The trainable variables in the
    Component will be trained once after calling this function.

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
      kl_divergence_name = Constants.MODEL_LOSS_KL_POSTFIX
      y_true.setdefault(
          kl_divergence_name ,
          self.z_prior_parameterizer(tf.ones((1, 1))))

      z_key = Constants.MODEL_OUTPUTS_Z
      z_params_key = Constants.MODEL_OUTPUTS_Z_PARAMS
        
      y_pred.setdefault(
          kl_divergence_name ,
          tf.concat(
              [results.get(z_key), results.get(z_params_key)],
              axis=-1))

      for modality_name in self.modality_names:
        negative_log_data_likelihood_name = f'{modality_name}/{Constants.MODEL_LOSS_DATA_POSTFIX}'
        modality_key = f"{modality_name}/{Constants.TENSOR_NAME_X}"
        data = ToDense(modality_key)(data)
        y_true.setdefault(
            negative_log_data_likelihood_name,
            data.get(modality_key))
        y_pred.setdefault(
            negative_log_data_likelihood_name,
            results.get(f"{modality_name}/{Constants.MODEL_OUTPUTS_X_PARAMS}"))

      loss = self.compiled_loss(y_true, y_pred)
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(y_true, y_pred)
    
    names = ['loss'] + [x.name for x in self.compiled_loss._losses]
    return {name: m.result() for name, m in zip(names, self.metrics)}

  def __setattr__(self, name: str, value: Any) -> None:
    """Overwrite __setattr__ function, so that everytime setting
    trainable to False, it automatically set alpha in the 
    progressive_scaler to 1.0.

    Parameters
    ----------
    name: str
        name of the attributes

    value: Any
        new value of the attributes.

    """
    super().__setattr__(name, value)
    if name == 'trainable':
      if not value:
        self.set_progressive_scaler_iteration(1, 1)
    
    return
     
  def set_progressive_scaler_iteration(
      self,
      current_iteration: int = 1,
      total_iterations: int = 1) -> None:
    """Set the alpha values (current iteration and total iterations) 
    for the hierarchical encoder.

    Parameters
    ----------
    current_iteration: int, optional
        current iteration. Defaults to 1.

    total_iterations: int, optional
        total iteartions, Defaults to 1.
    """
    total_iterations = float(total_iterations)
    current_iteration = float(current_iteration)
    progressive_scaler = self.hierarchical_encoder.progressive_scaler
    progressive_scaler.total_iterations.assign(total_iterations)
    progressive_scaler.current_iteration.assign(current_iteration)
    
    return