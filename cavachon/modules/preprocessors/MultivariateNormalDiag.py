from cavachon.layers.modifiers.DuplicateMatrix import DuplicateMatrix
from cavachon.layers.modifiers.LogTransform import LogTransform
from cavachon.layers.modifiers.NormalizeLibrarySize import NormalizeLibrarySize
from cavachon.layers.modifiers.ToDense import ToDense

import functools
import tensorflow as tf

class MultivariateNormalDiag(tf.keras.Model):
  def __init__(self, key):
    super().__init__()
    self.key = key
    self.modifiers = [
      ToDense(key),
      DuplicateMatrix(key)
    ]
  
  def call(self, inputs, training=None, mask=None):
    modifiers = self.modifiers
    if not training:
      modifiers = self.modifiers[:1] + self.modifiers[2:]

    return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)