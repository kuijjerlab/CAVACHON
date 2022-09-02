from cavachon.environment.Constants import Constants
from typing import Iterable, MutableMapping

import tensorflow as tf

class NormalizeLibrarySize(tf.keras.layers.Layer):
  """NormalizedLibrarySize

  Modifier used to normalize the tf.Tensor with library size (the sum 
  of values in the last dimension)

  Attributes
  ----------
  key: Any
      key to access the data needed to be normalized.

  """
  def __init__(self, key: Iterable, *args, **kwargs):
    """Constructor for NormalizeLibrarySize

    Parameters
    ----------
    key: Any
        key to access the data needed to be normalized.

    args:
        additional parameters for tf.keras.layers.Layer

    kwargs: Mapping[str, Any]
        additional parameters for tf.keras.layers.Layer

    """
    super().__init__(*args, **kwargs)
    self.key = key

  def call(
      self,
      inputs: MutableMapping[Iterable, tf.Tensor]) -> MutableMapping[Iterable, tf.Tensor]:
    """Normalize tf.Tensor stored in input with library size (the sum 
    of values in the last dimension)

    Parameters
    ----------
    inputs: MutableMapping[Any, tf.Tensor])
        inputs MutableMapping of tf.Tensor contains self.key

    Returns
    -------
    MutableMapping[Any, tf.Tensor]
        processed MutableMapping of tf.Tensor, where the library size 
        used to normalize the data will be stored in
        (self.key, 'libsize'). This can be used to scale back to the
        original data.

    """
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    libsize = tf.expand_dims(tf.reduce_sum(tensor, axis=-1), -1)
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