from cavachon.modifier.batch.BatchModifier import BatchModifier
from typing import Dict

import tensorflow as tf
import warnings

class Binarize(BatchModifier):

  def __init__(self, threshold: float = 1.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if threshold > 1.0 or threshold < 0.0:
      message = ''.join((
          f'Invalid range for threshold in {self.__class__.__name___} ',
          f'({self.name}). Expected 0 ≤ threshold ≤ 1, get {threshold}. Set to 1.0.'
      ))
      warnings.warn(message, RuntimeWarning)
      threshold = 1.0
    self.threshold = threshold
    
  def process(self, batch: Dict[str, tf.Tensor], selector: str) -> tf.Tensor:
    is_sparse = False
    tensor = batch.get(selector)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    tensor = tf.where(tensor >= self.threshold, tf.ones_like(tensor), tensor)
    tensor = tf.where(tensor < self.threshold, tf.zeros_like(tensor), tensor)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)
    return tensor