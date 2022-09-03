from cavachon.modules.parameterizers.Parameterizer import Parameterizer

class IndependentBernoulli(Parameterizer):
  """IndependentBernoulli

  Parameterizer for IndependentBernoulli. By defaults, the call() 
  function expects a Mapping of tf.Tensor with 'input'. The call() 
  function generate one single tf.Tensor which can be considered as the
  logits for IndependentBernoulli distribution.

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
  distributions.IndependentBernoulli: the compatible distribution
  layers.parameterizer.IndependentBernoulli: the parameterizer layer.
  modules.parameterizer.Parameterizer: the parent class.
  
  """
  default_libsize_scaling = False

  def __init__(self, *args, **kwargs):
    """Constructor for IndependentBernoulli. Should not be called 
    directly most of the time. Please use make() to create the model.

    Parameters
    ----------
    args: Any
        parameters used to initialize IndenpendentBernoulli.
    
    kwargs: Mapping[str, Any]
        parameters used to initialize IndenpendentBernoulli.

    """
    super().__init__(*args, **kwargs)

  @classmethod
  def make(
      cls,
      input_dims: int,
      event_dims: int,
      name: str = 'independent_bernoulli',
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
        Name for the tensorflow model. Defaults to 'independent_bernoulli'.

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
      