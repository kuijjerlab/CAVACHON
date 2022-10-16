from typing import Any, MutableMapping

import tensorflow as tf
import warnings

class Binarize(tf.keras.layers.Layer):
  """Binarize
  
  Modifier used to binarize the tf.Tensor stored in a MutableMapping.

  Attributes
  ----------
  key: Any
      key to access the data needed to be binarized.

  threshold: float
      threshold used to binarize the tf.Tensor.

  """
  def __init__(self, key: Any, threshold: float = 1.0, *args, **kwargs):
    """Constructor for Binarize

    Parameters
    ----------
    key: Any
        key to access the data needed to be binarized.
    
    threshold: float, optional
        threshold used to binarize the data. If the value is larger
        than this threshold, it will be set to 1.0. Needs to be in the 
        range of [0.0, 1.0] Defaults to 1.0.


    args:
        additional parameters for tf.keras.layers.Layer

    kwargs: Mapping[str, Any]
        additional parameters for tf.keras.layers.Layer

    """
    super().__init__(*args, **kwargs)
    if threshold > 1.0 or threshold < 0.0:
      message = ''.join((
          f'Invalid range for threshold in {self.__class__.__name___} ',
          f'({self.name}). Expected 0 ≤ threshold ≤ 1, get {threshold}. Set to 1.0.'
      ))
      warnings.warn(message, RuntimeWarning)
      threshold = 1.0
    self.threshold = threshold
    self.key = key
    
  def call(self, inputs: MutableMapping[Any, tf.Tensor]) -> MutableMapping[Any, tf.Tensor]:
    """Binarize to tf.Tensor stored in inputs.

    Parameters
    ----------
    inputs: MutableMapping[Any, tf.Tensor])
        inputs MutableMapping of tf.Tensor contains self.key

    Returns
    -------
    MutableMapping[Any, tf.Tensor]
        processed MutableMapping of tf.Tensor
    """
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    tensor = tf.where(tensor >= self.threshold, tf.ones_like(tensor), tensor)
    tensor = tf.where(tensor < self.threshold, tf.zeros_like(tensor), tensor)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    inputs[self.key] = tensor
    return inputs