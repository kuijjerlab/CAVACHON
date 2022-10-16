from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf
import warnings

class ScaleLibrarySize(tf.keras.layers.Layer):
  """TODO"""
  def __init__(self, key: Any, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.show_warning = True
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    
    if isinstance(self.key, str):
      libsize_key = f'{self.key}/{Constants.TENSOR_NAME_LIBSIZE}'
    else:
      libsize_key = self.key[:-1] + (Constants.TENSOR_NAME_LIBSIZE, )

    if self.show_warning and libsize_key not in inputs:
      message = ''.join((
        f'{libsize_key} is not in batch, ignore process in ',
        f'{self.__class__.__name__}. Please use NormalizeLibrarySize ',
        f'in preprocessing.'
      ))
      warnings.warn(message, RuntimeWarning)
      self.show_warning = False
    libsize = inputs.get(libsize_key, tf.ones((tensor.shape[0], 1)))
    tensor = tensor * libsize
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    inputs[self.key] = tensor
    inputs[libsize_key] = libsize
    return inputs