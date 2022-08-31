from cavachon.distributions.Distribution import Distribution
from typing import Mapping, Union

import tensorflow as tf
import tensorflow_probability as tfp

class IndependentZeroInflatedNegativeBinomial(Distribution, tfp.distributions.Mixture):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return
  
  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    if isinstance(params, tf.Tensor):
      logits, mean, dispersion = tf.split(params, 3, axis=-1)
    elif isinstance(params, Mapping):
      logits = params.get('logits')
      mean = params.get('mean')
      dispersion = params.get('dispersion')
    
    probs = tf.math.sigmoid(logits)
    probs = tf.stack([probs, 1 - probs], axis=-1)

    # batch_shape: (batch, ), event_shape: (event_dims, )
    return cls(
        cat=tfp.distributions.Categorical(probs=probs),
        components=(
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                1e-7 * tf.ones_like(mean),
                dispersion),
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                mean,
                dispersion)),
        **kwargs)