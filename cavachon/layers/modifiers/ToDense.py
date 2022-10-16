from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf

class ToDense(tf.keras.layers.Layer):
  """ToDense
  
  Modifier used to convert tf.SparseTensor stored in a MutableMapping 
  to dense tf.Tensor.

  Attributes
  ----------
  key: Any
      key to access the data needed to be transform to dense tf.Tensor.

  """
  def __init__(self, key: Any, *args, **kwargs):
    """Constructor for ToDense

    Parameters
    ----------
    key: Any
        key to access the data needed to be transformed.

    """
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    """Transform tf.SparseTensor stored in inputs to tf.Tensor.

    Parameters
    ----------
    inputs: MutableMapping[Any, tf.Tensor])
        inputs MutableMapping of tf.Tensor contains self.key

    Returns
    -------
    MutableMapping[Any, tf.Tensor]
        processed MutableMapping of tf.Tensor

    """
    
    tensor = inputs.get(self.key)
    if isinstance(tensor, tf.SparseTensor):
      tensor = tf.sparse.to_dense(tensor)
    inputs[self.key] = tensor
    return inputs