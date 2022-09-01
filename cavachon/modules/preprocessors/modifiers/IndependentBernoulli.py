from cavachon.environment.Constants import Constants
from cavachon.layers.modifiers.DuplicateMatrix import DuplicateMatrix
from cavachon.layers.modifiers.Binarize import Binarize
from cavachon.layers.modifiers.ToDense import ToDense

import functools
import tensorflow as tf

class IndependentBernoulli(tf.keras.Model):
  def __init__(self, modality_name):
    super().__init__()
    self.modality_name = modality_name
    self.modality_key = (modality_name, Constants.TENSOR_NAME_X)
    self.modifiers = [
      ToDense(self.modality_key),
      Binarize(self.modality_key)
    ]
  
  def call(self, inputs, training=None, mask=None):
    modifiers = self.modifiers
    return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)