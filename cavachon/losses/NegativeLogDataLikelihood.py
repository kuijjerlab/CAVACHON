from cavachon.distributions.Distribution import Distribution
from cavachon.utils.ReflectionHandler import ReflectionHandler
from typing import Union

import tensorflow as tf

class NegativeLogDataLikelihood(tf.keras.losses.Loss):
  def __init__(
      self,
      dist_x_z: Union[str, Distribution],
      name: str = 'NegativeLogDataLikelihood',
      **kwargs):
    super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs)
    if isinstance(dist_x_z, str):
      dist_x_z = ReflectionHandler.get_class_by_name(dist_x_z, 'distributions')
    self.dist_x_z_class = dist_x_z

  def call(self, y_true, y_pred):
    dist_x_z = self.dist_x_z_class.from_parameterizer_output(y_pred)
    logpx_z = tf.reduce_sum(dist_x_z.log_prob(y_true), axis=-1)

    return -logpx_z