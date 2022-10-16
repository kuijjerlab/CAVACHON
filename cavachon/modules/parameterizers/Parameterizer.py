from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Mapping

import tensorflow as tf

class Parameterizer(tf.keras.Model):
  """Parameterizer

  General Parameterizer modules, the difference between 
  modules/parameterizers and layers/parameterizers is that the 
  Parameterizers implemented in modules allows multiple inputs and
  outputs which uses the functional API of Tensorflow. This allows 
  postprocessing to the parameters (e.g. scale the mean by libsize).
  This class is used in the modules/components and modules/Model. Note 
  that, most of the modules/parameterizers uses the layers implemented 
  in layers/parameterizers but this is not required. It is possible to 
  simply use the modules/parameterizers if developers want to make 
  custom Parameterizer. By defaults, the call() function expects a 
  Mapping of tf.Tensor with 'input' (and 'libsize' if libsize_scaling 
  is set to True) and other custom keys. The call() function generate
  one single tf.Tensor as outputs.

  Attributes
  ----------
  default_libsize_scaling: bool
      class attributes, whether to scale by libsize by default (this
      is used when building tf.keras.Model to understand if the module
      requires multiple inputs.)
    
  libsize_scaling: bool
      whether to perform scaling by libsize.

  exp_transform: bool
      whether to perform exponential transformation. This will be 
      performed after scaling by libsize by default.

  """
  default_libsize_scaling = False

  def __init__(
      self,
      inputs: Mapping[str, tf.Tensor],
      outputs: tf.Tensor,
      name: str = 'parameterizer',
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs):
    """Constructor for Parameterizer. Should not be called directly 
    most of the time. Please use make() to create the model.

    Parameters
    ----------
    inputs: Mapping[str, tf.Tensor]
        inputs for building tf.keras.Model using Tensorflow functional 
        API. Expect to have key 'input' by default.
        
    outputs: tf.Tensor
        outputs for building tf.keras.Model using Tensorflow functional
        API.
    
    name: str, optional
        Name for the tensorflow model. Defaults to  'parameterizer'.
    
    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to False.
    
    exp_transform: bool, optional
        whether to perform exponential transformation. This will be 
        performed after scaling by libsize by default. Defaults to 
        False.
    
    kwargs: Mapping[str, Any]
        additional parameters used to initialize custom Parameterizer.

    """
    super().__init__(inputs=inputs, outputs=outputs, name=name)
    self.libsize_scaling = libsize_scaling
    self.exp_transform = exp_transform
  
  @classmethod
  def setup_inputs(
      cls,
      input_dims: int,
      libsize_scaling: bool = False,
      **kwargs) -> Mapping[str, tf.keras.Input]:
    """Builder function to setup the inputs tensors using Tensorflow 
    functional API. Developers can overwrite this function to create 
    custom Parameterizer.
        
    Parameters
    ----------
    input_dims: int
        input tf.Tensor dimension
    
    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to False.

    Returns
    -------
    Mapping[str, tf.keras.Input]:
        Inputs for building tf.keras.Model using Tensorflow functional 
        API.
    
    """
    inputs = dict()
    inputs.setdefault(Constants.TENSOR_NAME_X, tf.keras.Input(shape=(input_dims, )))
    if libsize_scaling:
      inputs.setdefault(Constants.TENSOR_NAME_LIBSIZE, tf.keras.Input(shape=(1, )))
    
    return inputs

  @classmethod
  def modify_outputs(
      cls,
      inputs: Mapping[str, tf.keras.Input],
      outputs: tf.Tensor, 
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs) -> tf.Tensor:
    """Postprocess the parameters created by layers.parameterizer. In
    particular, it allows libsize scaling and exponential transform.
    Developers can overwrite this function to create custom 
    Parameterizer.
        
    Parameters
    ----------
    input_dims: int
        input tf.Tensor dimension
    
    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to False.

    exp_transform: bool, optional
        whether to perform exponential transformation. This will be 
        performed after scaling by libsize by default. Defaults to 
        False.
    
    kwargs: Mapping[str, Any]
        additional parameters to modify the outputs, used for custom
        modify_outputs() function.

    Returns
    -------
    Mapping[str, tf.keras.Input]:
        Inputs for building tf.keras.Model using Tensorflow functional 
        API.
    
    """

    if libsize_scaling:
      outputs *= inputs.get(Constants.TENSOR_NAME_LIBSIZE)
    if exp_transform:
      outputs = tf.math.exp(outputs)
    
    return outputs

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'parameterizer',
      libsize_scaling: bool = False,
      exp_transform: bool = False,
      **kwargs):
    """Make the tf.keras.Model using the functional API of Tensorflow.
        
    Parameters
    ----------
    input_dims: int
        input tf.Tensor dimension. By default, it should be the last
        dimension of the outputs from previous layer.
    
    event_dims: int
        number of event dimensions for the outputs distribution (not
        the actuall outputs dimensions, which will be 
        event_dims * n_parameters)

    name: str, optional
        Name for the tensorflow model. Defaults to 'parameterizer'.

    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to False.

    exp_transform: bool, optional
        whether to perform exponential transformation. This will be 
        performed after scaling by libsize by default. Defaults to 
        False.
    
    kwargs: Mapping[str, Any]
        additional parameters, used for custom setup_inputs(), building
        parameterizer layer and modify_outputs() function.

    Returns
    -------
    tf.keras.Model
        Created model using Tensorflow functional API.
    
    """
    
    inputs = cls.setup_inputs(input_dims, libsize_scaling, **kwargs)
    layer_class = ReflectionHandler.get_class_by_name(cls.__name__, 'layers/parameterizers')
    layer = layer_class(event_dims=event_dims, name='parameterizer', **kwargs)
    outputs = layer(inputs.get(Constants.TENSOR_NAME_X))
    outputs = cls.modify_outputs(
        inputs=inputs,
        outputs=outputs,
        libsize_scaling=libsize_scaling,
        exp_transform=exp_transform,
        **kwargs)

    return cls(
        inputs=inputs,
        outputs=outputs,
        name=name,
        libsize_scaling=libsize_scaling,
        exp_transform=exp_transform)

