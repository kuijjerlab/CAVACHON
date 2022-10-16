from cavachon.distributions.Distribution import Distribution
from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class MixtureMultivariateNormalDiag(Distribution, tfp.distributions.MixtureSameFamily):
  """MixtureMultivariateNormalDiag
    
  Distribution for mixture of multivarate normal distributions with 
  diagonal covariance matrix (mainly used for the priors of latent 
  distributions).

  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    """Create mixture of multivarate normal distributions with diagonal 
    covariance matrix from the outputs of 
    modules.parameterizers.MixtureMultivariateNormalDiag.

    Parameters
    ----------
    params: Union[tf.Tensor, Mapping[str, tf.Tensor]]
        Parameters for the distribution created by parameterizers. 
        Alternatively, a mapping of tf.Tensor with parameter name as 
        keys can be provided. If provided with a tf.Tensor. The last 
        dimension needs to be a multiple of 2 plus 1,, and:
        1. params[..., component, 0] will be used as the mixture logits.
        2. params[..., component, 1:p+1] will be used as the loc (mean). 
        3. params[..., component, p+1:] will be used as the scale_diag 
           (std).
        If provided with a Mapping, 'logits', 'loc' and 'scale_diag' 
        should be in the keys of the Mapping. Note that
        1. The batch_shape should be the shape exclude the last two
           dimensions of the provided tf.Tensor. For instance if provided
           params.shape: (3, 5, 11). The batch_shape will be (3).
        2. The event_shape should be half of the last dimension ([p]) 
           of the provided tf.Tensor.
        3. The number of components will be the second last dimension.
           For instance if provided params.shape: (3, 5, 11). The number
           of components will be 5.

    Returns
    -------
    tfp.distributions.Distribution
        Created Tensorflow Probability MixtureMultivariateNormalDiag
        Distribution.
    
    """
    if isinstance(params, tf.Tensor):
      # shape: (batch, n_components)
      logits = params[..., 0]
      # shape: (batch, n_components, event_dims * 2)
      components_params = params[..., 1:]
      # batch_shape: (batch, n_components), event_shape: (event_dims, )
      components_distribution = MultivariateNormalDiag.from_parameterizer_output(components_params)
    elif isinstance(params, Mapping):
      logits = params.get('logits')
      loc = params.get('loc')
      scale_diag = params.get('scale_diag')
      components_distribution = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
    
    # batch_shape: (batch, ), event_shape: (, )
    mixture_distribution = tfp.distributions.Categorical(logits=logits)
    
    # batch_shape: (batch, ), event_shape: (event_dims, )
    return cls(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution,
        **kwargs)