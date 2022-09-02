from typing import Any, MutableMapping

import tensorflow as tf

class ExpTransform(tf.keras.layers.Layer):
  """ExpTransform
  
  Modifier used to exponential transform the tf.Tensor stored in a 
  MutableMapping.

  Attributes
  ----------
  key: Any
      key to access the data needed to be exponential transformed.

  """
  def __init__(self, key: Any, *args, **kwargs):
    """Constructor for ExpTransform

    Parameters
    ----------
    key: Any
        key to access the data needed to be exponential transformed.

    """
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[Any, tf.Tensor]) -> MutableMapping[Any, tf.Tensor]:
    """Exponential transform tf.Tensor stored in inputs.

    Parameters
    ----------
    inputs: MutableMapping[Any, tf.Tensor])
        inputs MutableMapping of tf.Tensor contains self.key

    Returns
    -------
    MutableMapping[Any, tf.Tensor]
        processed MutableMapping of tf.Tensor
    """
    is_sparse = False
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
      is_sparse = True
    tensor = tf.math.exp(tensor)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    inputs[self.key] = tensor
    return inputs