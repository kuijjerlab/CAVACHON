from cavachon.distributions.Distribution import Distribution
from typing import Union, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

class MultivariateNormalDiag(Distribution, tfp.distributions.MultivariateNormalDiag):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def from_parameterizer_output(
      cls,
      params: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      **kwargs):
    if isinstance(params, tf.Tensor):
      loc, scale_diag = tf.split(params, 2, axis=-1)
    elif isinstance(params, Mapping):
      loc = params.get('loc')
      scale_diag = params.get('scale_diag')

    # batch_shape: (batch, ), event_shape: (event_dims, )
    return cls(loc=loc, scale_diag=scale_diag, **kwargs)

