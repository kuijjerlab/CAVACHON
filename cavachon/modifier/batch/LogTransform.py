from cavachon.modifier.batch.BatchModifier import BatchModifier
from typing import Dict

import tensorflow as tf

class LogTransform(BatchModifier):

  def __init__(self, pseudocount: float = 1.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pseudocount = pseudocount
  
  def process(self, batch: Dict[str, tf.Tensor], selector: str) -> tf.Tensor:
    is_sparse = False
    tensor = batch.get(selector)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    target = tf.math.log(target + self.pseudocount)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)
    return tensor