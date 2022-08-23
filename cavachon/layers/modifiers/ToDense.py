from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf

class ToDense(tf.keras.layers.Layer):

  def __init__(self, key: Any, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
    inputs[self.key] = tensor
    return inputs