from typing import Any, MutableMapping

import tensorflow as tf

class LogTransform(tf.keras.layers.Layer):
  """LogTransform
  
  Modifier used to log-transform the tf.Tensor stored in a 
  MutableMapping.

  Attributes
  ----------
  pseudocount: float
      values added to the tf.Tensor before log-transformation (to avoid
      -inf outcome)

  key: Any
      key to access the data needed to be log-transformed.

  """
  def __init__(self, key: Any, pseudocount: float = 1.0, *args, **kwargs):
    """Constructor for LogTransform

    Parameters
    ----------
    key: Any
        key to access the data needed to be binarized.

    pseudocount: float, optional
        values added to the tf.Tensor before log-transformation (to 
        avoid -inf outcome)

    args:
        additional parameters for tf.keras.layers.Layer

    kwargs: Mapping[str, Any]
        additional parameters for tf.keras.layers.Layer

    """
    super().__init__(*args, **kwargs)
    self.pseudocount = pseudocount
    self.key = key
  
  def call(self, inputs: MutableMapping[Any, tf.Tensor]) -> MutableMapping[Any, tf.Tensor]:
    """Log-transform tf.Tensor stored in inputs.

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
    tensor = tf.math.log(tensor + self.pseudocount)
    if is_sparse:
      tensor = tf.sparse.from_dense(tensor)

    inputs[self.key] = tensor
    return inputs