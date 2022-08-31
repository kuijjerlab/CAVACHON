from cavachon.distributions.Distribution import Distribution
from typing import Mapping, Union

import tensorflow as tf
import tensorflow_probability as tfp

class IndependentBernoulli(Distribution, tfp.distributions.Bernoulli):
  def __init__(self, *args, **kwargs):
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

    return distribution