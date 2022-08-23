from cavachon.distributions.Distribution import Distribution
from cavachon.distributions.MultivariateNormalDiag import MultivariateNormalDiag
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class MixtureMultivariateNormalDiag(Distribution, tfp.distributions.MixtureSameFamily):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    if isinstance(params, tf.Tensor):
      # shape: (batch, n_components)
      logits = params[:, :, 0]
      # shape: (batch, n_components, event_dims * 2)
      components_params = params[:, :, 1:]
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