from cavachon.distributions.Distribution import Distribution
from typing import Mapping, Union

import tensorflow as tf
import tensorflow_probability as tfp

class IndependentBernoulli(Distribution, tfp.distributions.Bernoulli):
  """IndependentBernoulli
  
  Distribution for independent Bernoulli.
  
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return
  
  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    """Create independent Bernoulli distributions from the outputs of
    modules.parameterizers.IndependentBernoulli.

    Parameters
    ----------
    params: Union[tf.Tensor, Mapping[str, tf.Tensor]]
        Parameters for the distribution created by parameterizers. 
        Alternatively, a mapping of tf.Tensor with parameter name as 
        keys can be provided. If provided with a tf.Tensor, it will be 
        used as the logits to create Bernoulli distribution. If provided 
        with a Mapping, 'logits' should be in the keys of the Mapping.
        Note that:
        1. The batch_shape should be the same as the provided tf.Tensor. 
        2. The event_shape should be []. 

    Returns
    -------
    tfp.distributions.Distribution
        Created Tensorflow Probability Bernoulli Distribution.
    
    """
    if isinstance(params, tf.Tensor):
      logits = params
    elif isinstance(params, Mapping):
      logits = params.get('logits')
    
    distribution = tfp.distributions.Bernoulli(logits=logits)

    return distribution