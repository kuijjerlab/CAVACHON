from __future__ import annotations
from collections import OrderedDict
from collections.abc import Iterator
from typing import Optional
import tensorflow as tf

class Parameterizer(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.parameterizer_networks = OrderedDict()

  def __getitem__(self, __k: str) -> tf.keras.Model:
    return self.parameterizer_networks[__k]

  def __setitem__(self, __k: str, __v: tf.keras.Model) -> None:
    self.parameterizer_networks[__k] = __v
    return
  
  def __delitem__(self, __k: str) -> None:
    del self.parameterizer_networks[__k]
    return

  def __iter__(self) -> Iterator[Parameterizer]:
    return iter(self.parameterizer_networks)

  def __len__(self) -> int:
    return len(self.parameterizer_networks)

  def get(self, __key, __default: Optional[None] = None):
    return self.parameterizer_networks.get(__key, __default)

  def setdefault(self, __key, __value):
    self.parameterizer_networks.setdefault(__key, __value)
    return

  def keys(self):
    return self.parameterizer_networks.keys()
  
  def values(self):
    return self.parameterizer_networks.values()

  def items(self):
    return self.parameterizer_networks.items()