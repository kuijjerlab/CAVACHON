from typing import Any, MutableMapping

import tensorflow as tf
import warnings

class Binarize(tf.keras.layers.Layer):

  def __init__(self, key: Any, threshold: float = 1.0, *args, **kwargs):
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