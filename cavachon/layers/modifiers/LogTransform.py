from typing import Any, MutableMapping

import tensorflow as tf

class LogTransform(tf.keras.layers.Layer):

  def __init__(self, key: Any, pseudocount: float = 1.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pseudocount = pseudocount
    self.key = key
  
  def call(self, inputs: MutableMapping[Any, tf.Tensor]) -> MutableMapping[Any, tf.Tensor]:
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    target = tf.math.log(target + self.pseudocount)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    inputs[self.key] = tensor
    return inputs