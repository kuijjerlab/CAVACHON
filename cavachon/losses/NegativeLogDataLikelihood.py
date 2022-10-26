from cavachon.utils.ReflectionHandler import ReflectionHandler

import tensorflow as tf

class NegativeLogDataLikelihood(tf.keras.losses.Loss):
  """NegativeLogDataLikelihood
  
  NegativeLogDataLikelihood computed with
  tfp.distributions.Distribution log_prob.
  """
  def __init__(
      self,
      dist_x_z: str,
      weight: float = 1.0,
      name: str = 'negative_log_data_likelihood',
      **kwargs):
    """Constructor for NegativeLogDataLikelihood

    Parameters
    ----------
    dist_x_z: str
        the name for the distributions implemented in 
        cavachon.distributions

    weight: float, optional
        the scaling factor for the loss. The output will be 
        weight * loss. Defaults to 1.0.

    name: str, optional
        name for the tf.keras.losses.Loss (will be used when reporting
        the loss during training_step in Component and Model).
        Defaults to 'negative_log_data_likelihood'.
    
    kwargs: Mapping[str, Any]
        additional parameters for tf.keras.losses.Loss

    """
    self.weight = weight
    super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs)
    if isinstance(dist_x_z, str):
      dist_x_z = ReflectionHandler.get_class_by_name(dist_x_z, 'distributions')
    self.dist_x_z_class = dist_x_z

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the NegativeLogDataLikelihood loss

    Parameters
    ----------
    y_true: tf.Tensor
        The outputs of modules.parameterizers, with a shape of 
        (batch_dims, event_dims * num_parameters). Note that this 
        special requirement is designed to follow the API 
        tf.keras.losses.Loss provides. Can be ignored if the developers 
        wish to use custom eager training. 
    
    y_pred: tf.Tensor
        The observed samples from the distribution, with a shape of 
        (batch_dims, event_dims) 

    Returns
    -------
    tf.Tensor:
        The computed NegativeLogDataLikelihood loss
    """
    dist_x_z = self.dist_x_z_class.from_parameterizer_output(y_pred)
    logpx_z = tf.reduce_sum(dist_x_z.log_prob(y_true), axis=-1)

    return -self.weight * logpx_z