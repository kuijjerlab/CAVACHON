from cavachon.environment.Constants import Constants
from typing import Iterable, MutableMapping

import tensorflow as tf

class NormalizeLibrarySize(tf.keras.layers.Layer):
  """TODO"""
  def __init__(self, key: Iterable, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.key = key

  def call(
      self,
      inputs: MutableMapping[Iterable, tf.Tensor]) -> MutableMapping[Iterable, tf.Tensor]:
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    libsize = tf.reduce_sum(tensor, axis=-1)
    tensor = tensor / libsize
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)
    
    if isinstance(self.key, str):
      libsize_key = f'{self.key}/{Constants.TENSOR_NAME_LIBSIZE}'
    else:
      libsize_key = self.key[:-1] + (Constants.TENSOR_NAME_LIBSIZE, )

    inputs[self.key] = tensor
    inputs[libsize_key] = libsize
    return inputs