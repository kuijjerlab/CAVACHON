#%%
from __future__ import annotations
from cavachon.batch.BatchSpec import BatchSpec
from collections.abc import Iterator, Mapping
from typing import Any, Dict, List, Optional

import tensorflow as tf

class Batch(Mapping):
  def __init__(self, tensors: List[tf.Tensor], names: List[str]):
    super().__init__()
    self.n_tensors: int = 0
    self._tensor: Optional[tf.Tensor] = None
    self._specs: Dict[str, BatchSpec] = dict()
    start = 0
    for tensor, name in zip(tensors, names):
      size = tensor.shape[-1]
      spec = BatchSpec(name, start, start + size)
      self._specs.setdefault(name, spec)
      self.n_tensors += 1
      start += size
    self._tensor: tf.Tensor = tf.concat(tensors, axis=-1)
    
    return

  def __iter__(self) -> Iterator[tf.Tensor]:
    return iter(self._tensor)

  def __getitem__(self, __k: str) -> tf.Tensor:
    spec = self._specs[__k]
    start = spec.start
    end = spec.end
    return self._tensor[:, start:end]

  def __len__(self) -> int:
    return self._tensor.shape[0]
  
  @property
  def tensor(self):
    return self._tensor

  @classmethod
  def from_dict(cls, tensor_dict: Dict[str, Any]) -> int:
    tensors = []
    names = []
    for name, tensor in tensor_dict.items():
      names.append(name)
      tensors.append(tensor)
    
    return cls(tensors, names)
# %%
