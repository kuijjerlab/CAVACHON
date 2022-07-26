from cavachon.environment.Constants import Constants
from cavachon.modifier.batch.BatchModifier import BatchModifier
from typing import Dict

import tensorflow as tf
import warnings

class ScaleLibrarySize(BatchModifier):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.show_warning = True

  def process(self, batch: Dict[str, tf.Tensor], selector: str) -> tf.Tensor:
    is_sparse = False
    tensor = batch.get(selector)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    
    selector_libsize = f"{self.selector}/{Constants.TENSOR_NAME_LIBSIZE}"
    if self.show_warning and selector_libsize not in batch:
      message = ''.join((
        f'{selector_libsize} is not in batch, ignore process in ',
        f'{self.__class__.__name__}. Please use NormalizeLibrarySize ',
        f'in preprocessing.'
      ))
      warnings.warn(message, RuntimeWarning)
      self.show_warning = False
    libsize = batch.get(selector_libsize, tf.ones((tensor.shape[0], 1)))
    tensor = tensor * libsize
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    return tensor