from cavachon.distributions.Distribution import Distribution
from typing import Mapping, Union

import tensorflow as tf
import tensorflow_probability as tfp

class IndependentBernoulli(Distribution, tfp.distributions.Independent):
  def __init__(self, *args, **kwargs):
    # by default,
    # the event shape of Bernoulli is (, ), change it to logits.shape[1:]
    # the batch shape of Bernoulli is logits.shape, change it to (logits.shape[0], )
    super().__init__(*args, **kwargs)
    return
  
  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):     
    if isinstance(params, tf.Tensor):
      logits = params
    elif isinstance(params, Mapping):
      logits = params.get('logits')
    
    distribution = tfp.distributions.Bernoulli(logits=logits)
    reinterpreted_batch_ndims = tf.size(distribution.batch_shape_tensor()) - 1
    
    # batch_shape: (batch, ), event_shape: (event_dims, )
    return cls(
        distribution=distribution,
        reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        **kwargs)