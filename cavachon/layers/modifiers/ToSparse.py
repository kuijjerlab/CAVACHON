from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf

class ToSparse(tf.keras.layers.Layer):

  def __init__(self, key: Any, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.Tensor):
      tensor = tf.sparse.from_dense(tensor)
    inputs[self.key] = tensor
    return inputs