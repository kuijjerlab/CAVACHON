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
      probs, mean, dispersion = tf.split(params, 3, axis=-1)
      probs = tf.stack([probs, 1 - probs], axis=-1)
    elif isinstance(params, Mapping):
      probs = params.get('probs')
      mean = params.get('mean')
      dispersion = params.get('dispersion')
      probs = tf.stack([probs, 1 - probs], axis=-1)

    return cls(
        cat=tfp.distributions.Categorical(probs=probs),
        components=(
            tfp.distributions.Deterministic(tf.zeros_like(mean)),
            tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(mean, dispersion)),
        **kwargs)