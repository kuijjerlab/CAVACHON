from cavachon.modules.parameterizers.Parameterizer import Parameterizer
from typing import Mapping

import tensorflow as tf

class IndependentZeroInflatedNegativeBinomial(Parameterizer):
  """IndependentZeroInflatedNegativeBinomial

  Parameterizer for IndependentZeroInflatedNegativeBinomial. By 
  defaults, the call() function expects a Mapping of tf.Tensor with 
  'input' and 'libsize'. The call() function generate one single 
  tf.Tensor which can be considered as the logits, mean and dispersion
  for IndependentZeroInflatedNegativeBinomial distribution, and:
  1. outputs[..., 0:p] will be used as the logits. 
  2. outputs[..., p:2*p] will be used as the mean.
  3. outputs[..., 2*p:3*p] will be used as the dispersion.

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

  See Also
  --------
  distributions.IndependentZeroInflatedNegativeBinomial
      the compatible distribution
  
  layers.parameterizer.IndependentZeroInflatedNegativeBinomial
      the parameterizer layer.
  
  modules.parameterizerIndependentZeroInflatedNegativeBinomial
      the parent class.
  
  """
  default_libsize_scaling = True

  def __init__(self, *args, **kwargs):
    """Constructor for IndependentZeroInflatedNegativeBinomial. Should 
    not be called directly most of the time. Please use make() to create
    the model.

    Parameters
    ----------
    args: Any
        parameters used to initialize 
        IndependentZeroInflatedNegativeBinomial.
    
    kwargs: Mapping[str, Any]
        parameters used to initialize 
        IndependentZeroInflatedNegativeBinomial.

    """
    super().__init__(*args, **kwargs)

  @classmethod
  def modify_outputs(
      cls,
      inputs: Mapping[str, tf.keras.Input],
      outputs: tf.Tensor, 
      libsize_scaling: bool = True,
      exp_transform: bool = True,
      **kwargs) -> tf.Tensor:
    """Postprocess the parameters created by 
    layers.IndependentZeroInflatedNegativeBinomial. In particular, it 
    allows libsize scaling and exponential transform. In addition, it
    checks if the parameters are valid.
        
    Parameters
    ----------
    input_dims: int
        input tf.Tensor dimension
    
    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to True.

    exp_transform: bool, optional
        whether to perform exponential transformation. This will be 
        performed after scaling by libsize by default. Defaults to 
        True.
    
    kwargs: Mapping[str, Any]
        additional parameters to modify the outputs, used for custom
        modify_outputs() function.

    Returns
    -------
    Mapping[str, tf.keras.Input]:
        Inputs for building tf.keras.Model using Tensorflow functional 
        API.
    
    """

    logits, mean, dispersion = tf.split(outputs, 3, axis=-1)
    if libsize_scaling:
      mean *= inputs.get('libsize')
    if exp_transform:
      mean = tf.where(mean > 15., 15. * tf.ones_like(mean), mean)
      mean = tf.math.exp(mean) - 1.0
    
    mean = tf.where(mean == 0, 1e-7 * tf.ones_like(mean), mean)

    return tf.concat([logits, mean, dispersion], axis=-1)

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'independent_zero_inflated_negative_binomial',
      libsize_scaling: bool = True,
      exp_transform: bool = True):
    """Make the tf.keras.Model using the functional API of Tensorflow.
        
    Parameters
    ----------
    input_dims: int
        input tf.Tensor dimension. By default, it should be the last
        dimension of the outputs from previous layer.
    
    event_dims: int
        number of event dimensions for the outputs distribution.
    
    name: str, optional
        Name for the tensorflow model. Defaults to
        'independent_zero_inflated_negative_binomial'.

    libsize_scaling: bool, optional
        whether to perform scaling by libsize. Defaults to False.

    exp_transform: bool, optional
        whether to perform exponential transformation. This will be 
        performed after scaling by libsize by default. Defaults to 
        False.
    
    Returns
    -------
    tf.keras.Model
        Created model using Tensorflow functional API.
    
    """
    return super().make(
      input_dims=input_dims,
      event_dims=event_dims,
      name=name,
      libsize_scaling=libsize_scaling,
      exp_transform=exp_transform)