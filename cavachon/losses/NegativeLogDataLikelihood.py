from cavachon.distributions.Distribution import Distribution

import tensorflow as tf

class NegativeLogDataLikelihood(tf.keras.losses.Loss):
  def __init__(self, dist_x_z_class: Distribution, name: str = 'NegativeLogDataLikelihood',**kwargs):
    super().__init__(name=name, **kwargs)
    self.dist_x_z_class = dist_x_z_class

  def call(self, y_true, y_pred):
    dist_x_z = self.dist_x_z_class.from_parameterizer_output(y_pred)
    logpx_z = tf.reduce_sum(dist_x_z.log_prob(y_true), axis=-1)

    return -logpx_z