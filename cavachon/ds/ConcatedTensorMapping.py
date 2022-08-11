#%%
from __future__ import annotations
from cavachon.ds.ConcatedTensorSpec import ConcatedTensorSpec
from collections.abc import Iterator, Mapping
from typing import Dict, Hashable, List, Optional

import tensorflow as tf

class ConcatedTensorMapping(Mapping):
  def __init__(self, tensors: List[tf.Tensor], keys: List[Hashable]):
    super().__init__()
    self.n_tensors: int = 0
    self.concated_tensor: Optional[tf.Tensor] = None
    self.specs: Dict[Hashable, ConcatedTensorSpec] = dict()

    start = 0
    for tensor, name in zip(tensors, keys):
      size = tensor.shape[-1]
      spec = ConcatedTensorSpec(name, start, start + size)
      self.specs.setdefault(name, spec)
      self.n_tensors += 1
      start += size
    self.concated_tensor: tf.Tensor = tf.concat(tensors, axis=-1)
    
    return

  def __repr__(self) -> str:
    message = f'{self.__class__.__name__}{{'
    for i, (__k, __v) in enumerate(self.items()):
      message += f'{__k}: {repr(__v)}'
      if i != len(self.specs) - 1:
        message += ', '
      else:
        message += '}'
    return message

  def __iter__(self) -> Iterator[ConcatedTensorMapping]:
    return iter(self.specs)

  def __getitem__(self, __k: str) -> tf.Tensor:
    spec = self.specs[__k]
    start = spec.start
    end = spec.end
    return self.concated_tensor[:, start:end]

  def __len__(self) -> int:
    return len(self.specs)

  @property
  def tensor(self):
    return self.concated_tensor

  @classmethod
  def from_mapping(cls, tensor_mapping: Mapping[Hashable, tf.Tensor]) -> ConcatedTensorMapping:
    tensors = []
    keys = []
    for key, tensor in tensor_mapping.items():
      keys.append(key)
      tensors.append(tensor)
    
    return cls(tensors, keys)

