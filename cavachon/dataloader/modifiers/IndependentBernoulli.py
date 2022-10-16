from cavachon.environment.Constants import Constants
from cavachon.layers.modifiers.Binarize import Binarize
from cavachon.layers.modifiers.ToDense import ToDense
from typing import Any, Mapping, Tuple

import functools
import tensorflow as tf

class IndependentBernoulli(tf.keras.Model):
  """IndependentBernoulli

  Modifiers for the modality which is IndependentBernoulli distribution.
  The instance will be used right after the tf.data.Dataset is created
  using the DataLoader.

  Attributes
  ----------
  modality_names: str
      modality name.

  modality_key: str
      the key used to access the mapping of data created from 
      tf.data.Dataset. Defaults to `modality_name`/matrix.

  modifiers: List[tf.keras.layers.Layer]
      list of modifiers that will be applied to the data created from 
      tf.data.Dataset. Defaults to [ToDense, Binarize].
  
  See Also
  --------
  DataLoader: used to create tf.data.Dataset from MuData.

  """
  def __init__(self, modality_name):
    """Constructor for IndependentBernoulli (modifier for 
    tf.data.Dataset)

    Parameters
    ----------
    modality_name: str 
        the name of modality that needs to be processed.
    """
    super().__init__()
    self.modality_name: str = modality_name
    self.modality_key: str = f"{modality_name}/{Constants.TENSOR_NAME_X}"
    self.modifiers = [
      ToDense(self.modality_key),
      Binarize(self.modality_key)
    ]
  
  def call(self, inputs: Mapping[Any, tf.Tensor], training=None, mask=None):
    """Processed the data created from tf.data.Dataset.

    Parameters
    ----------
    inputs: 
        Mapping of tf.Tensor, where the keys contain self.modality_key. 

    training: bool, optional
        Not used (kept for tf.keras.Model API). 

    mask: tf.Tensor, optional
        Not used (kept for tf.keras.Model API). 

    Returns
    -------
    Mapping[Any, tf.Tensor]
        processed data. 

    """
    modifiers = self.modifiers
    return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)