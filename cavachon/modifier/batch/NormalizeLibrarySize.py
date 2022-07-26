from cavachon.environment.Constants import Constants
from cavachon.modifier.batch.BatchModifier import BatchModifier
from typing import Dict, Tuple

import tensorflow as tf

class NormalizeLibrarySize(BatchModifier):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def process(self, batch: Dict[str, tf.Tensor], selector: str) -> Tuple[tf.Tensor, tf.Tensor]:
    is_sparse = False
    tensor = batch.get(selector)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    libsize = tf.reduce_sum(tensor, axis=-1)
    tensor = tensor / libsize
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    return tensor, libsize

  def execute(
      self,
      batch: Dict[str, tf.Tensor],
      selector: str) -> Dict[str, tf.Tensor]:
    new_tensor, libsize = self.process(batch)
    batch = self.complete(new_tensor, libsize, batch, selector)
    return batch

  def complete(
      self,
      new_tensor: tf.Tensor,
      libsize: tf.Tensor,
      batch: Dict[str, tf.Tensor],
      selector: str
  ) -> Dict[str, tf.Tensor]:
    if self.keep_original:
      batch.setdefault(f'{selector}/original', batch[f'{selector}'])
    else:
      batch[f'{selector}/original'] = new_tensor
    batch[f'{selector}'] = new_tensor
    batch[f'{selector}/{Constants.TENSOR_NAME_LIBSIZE}'] = libsize
    return batch