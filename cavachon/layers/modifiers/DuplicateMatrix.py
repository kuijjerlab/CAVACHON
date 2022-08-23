from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf

class DuplicateMatrix(tf.keras.layers.Layer):

  def __init__(self, key: Any, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    
    if isinstance(self.key, str):
      duplicate_key = f'{self.key}/{Constants.TENSOR_NAME_ORIGIONAL_X}'
    else:
      duplicate_key = self.key[:-1] + (Constants.TENSOR_NAME_ORIGIONAL_X, )

    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)
    inputs[duplicate_key] = tensor

    return inputs