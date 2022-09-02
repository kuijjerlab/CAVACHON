from cavachon.distributions.Distribution import Distribution
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class MultivariateNormalDiag(Distribution, tfp.distributions.MultivariateNormalDiag):
  """MultivariateNormalDiag
  
  Distribution for multivarate normal distributions with 
  diagonal covariance matrix (mainly used for latent distributions).
  
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    """Create multivarate normal distributions with diagonal covariance 
    matrix from the outputs of 
    modules.parameterizers.MultivariateNormalDiag.

    Parameters
    ----------
    params: Union[tf.Tensor, Mapping[str, tf.Tensor]]
        Parameters for the distribution created by parameterizers. 
        Alternatively, a mapping of tf.Tensor with parameter name as 
        keys can be provided. If provided with a tf.Tensor. The last 
        dimension needs to be a multiple of 2,, and:
        1. params[..., 0:p] will be used as the loc (mean). 
        2. params[..., p:2*p] will be used as the scale_diag (std).
        If provided with a Mapping, 'loc' and 'scale_diag' should be in 
        the keys of the Mapping. Note that
        1. The batch_shape should be the shape exclude the last 
           dimension of the provided tf.Tensor. For instance if provided
           params.shape: (3, 5, 10). The batch_shape will be (3, 5).
        2. The event_shape should be half of the last dimension ([p]) 
           of the provided tf.Tensor.

    Returns
    -------
    tfp.distributions.Distribution
        Created Tensorflow Probability MultivariateNormalDiag 
        Distribution.
    
    """
    if isinstance(params, tf.Tensor):
      loc, scale_diag = tf.split(params, 2, axis=-1)
    elif isinstance(params, Mapping):
      loc = params.get('loc')
      scale_diag = params.get('scale_diag')

    # batch_shape: (batch, ), event_shape: (event_dims, )
    return cls(loc=loc, scale_diag=scale_diag, **kwargs)

