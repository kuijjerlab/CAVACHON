from cavachon.modules.parameterizers.Parameterizer import Parameterizer

class MixtureMultivariateNormalDiag(Parameterizer):
  """MixtureMultivariateNormalDiag

  Parameterizer for MixtureMultivariateNormalDiag. By defaults, 
  the call() function expects a Mapping of tf.Tensor with 'input'. 
  The call() function generate one single tf.Tensor which can be 
  considered as the logits, loc and scale_diag for 
  MixtureMultivariateNormalDiag distribution, and
  1. outputs[..., component, 0] will be used as the mixture logits.
  2. outputs[..., component, 1:p+1] will be used as the loc (mean). 
  3. outputs[..., component, p+1:] will be used as the scale_diag (std).

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
  distributions.MixtureMultivariateNormalDiag:
      the compatible distribution
  
  layers.parameterizer.MixtureMultivariateNormalDiag:
      the parameterizer layer.
  
  
  modules.parameterizer.MixtureMultivariateNormalDiag:
      the parent class.
  
  """

  default_libsize_scaling = False

  def __init__(self, *args, **kwargs):
    """Constructor for MultivariateNormalDiag. Should not be called 
    directly most of the time. Please use make() to create the model.

    Parameters
    ----------
    args: Any
        parameters used to initialize MixtureMultivariateNormalDiag.
    
    kwargs: Mapping[str, Any]
        parameters used to initialize MixtureMultivariateNormalDiag.

    """
    super().__init__(*args, **kwargs)

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      n_components: int,
      name: str = 'mixture_multivariate_normal_diag',
      libsize_scaling: bool = False,
      exp_transform: bool = False):
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
        'mixture_multivariate_normal_diag'.

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
      n_components=n_components,
      name=name,
      libsize_scaling=libsize_scaling,
      exp_transform=exp_transform)