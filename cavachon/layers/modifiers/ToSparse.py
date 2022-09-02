from cavachon.environment.Constants import Constants
from typing import Any, MutableMapping

import tensorflow as tf

class ToSparse(tf.keras.layers.Layer):
  """ToDense
  
  Modifier used to convert tf.Tensor stored in a MutableMapping to 
  tf.SparseTensor.

  Attributes
  ----------
  key: Any
      key to access the data needed to be transform to tf.SparseTensor.

  """
  def __init__(self, key: Any, *args, **kwargs):
    """Constructor for ToSparse

    Parameters
    ----------
    key: Any
        key to access the data needed to be transformed.

    """
    super().__init__(*args, **kwargs)
    self.key = key

  def call(self, inputs: MutableMapping[str, tf.Tensor]) -> MutableMapping[str, tf.Tensor]:
    """Transform tf.Tensor stored in inputs to tf.SparseTensor.

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
    if isinstance(tensor, tf.Tensor):
      tensor = tf.sparse.from_dense(tensor)
    inputs[self.key] = tensor
    return inputs