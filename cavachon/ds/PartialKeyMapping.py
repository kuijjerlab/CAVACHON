from __future__ import annotations
from types import ClassMethodDescriptorType

from importlib_metadata import Pair
from cavachon.ds.KeyNode import PartialKeyNode, FullKeyNode, KeyNode
from collections.abc import MutableMapping, Iterator
from typing import Any, Dict, Iterable, Hashable, Mapping

class PartialKeyMappingIterator(Iterator):
  def __init__(self, mapping: PartialKeyMapping):
    self.mapping = mapping
    self.iterator = iter(mapping.nodes)

  def __iter__(self) -> PartialKeyMappingIterator:
    return self

  def __next__(self) -> Any:
    while True:
      key = next(self.iterator)
      node = self.mapping.nodes.get(key)
      if isinstance(node, FullKeyNode):
        if len(key) == 1:
          return key[0]
        else:
          return key

class PartialKeyMapping(MutableMapping):
  def __init__(self, *args, **kwargs):
    self.nodes: Dict[Hashable, KeyNode] = dict()
    self.n = 0

    if len(args) > 1:
      raise TypeError(f'{self.__class__.__name__}: expected at most 1 argument, got {len(args)}')
    self.update(*args, **kwargs)
  
  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.nodes})'
  
  def __setitem__(self, __k: Hashable, __v: Any) -> None:
    if isinstance(__k, str):
      __k = (__k, )

    # If the key existed in the mapping, simply change the value.
    if __k in self.nodes:
      node = self.nodes.get(__k)
      if isinstance(node, PartialKeyNode):
        raise AttributeError(
            f'{self.__class__.__name__}: attempt to assign value to {node.__class__.__name__}')
      node.value = __v
      return

    # Otherwise create the PartialKeyNode if necessary, add a new FullKeyNode and update n_values
    # along the path.
    node = None
    for i in range(len(__k)):
      partial_key = __k[:i+1]
      if partial_key in self.nodes:
        next_node = self.nodes.get(partial_key)
      else:
        if i == len(__k) - 1:
          next_node = FullKeyNode(__k, __v)
        else:
          next_node = PartialKeyNode(partial_key)

        self.nodes[partial_key] = next_node
        if node is not None:
          node_key_layer_id = len(node.key)
          node.value[partial_key[node_key_layer_id:]] = next_node
      next_node.n_values += 1
      node = next_node
    
    self.n += 1
    return

  def __getitem__(self, __k: Hashable) -> Any:
    if isinstance(__k, str):
      __k = (__k, )

    node = self.nodes[__k]
    if isinstance(node, FullKeyNode):
      return node.value
    else:
      result = dict()
      node_key_layer_id = len(node.key)
      values_in_subtree = node.values_in_subtree
      for full_key, value in values_in_subtree.items():
        simplified_key = full_key[node_key_layer_id:]
        if len(simplified_key) == 1:
          simplified_key = simplified_key[0]
        result.setdefault(simplified_key, value)
      return PartialKeyMapping(result)

  def __delitem__(self, __k: Hashable) -> None:
    if isinstance(__k, str):
      __k = (__k, )

    # Delete everything below the nodes
    node = self.nodes[__k]
    n_remove_nodes = node.n_values
    values_in_subtree = node.values_in_subtree
    for full_key in values_in_subtree.keys():
      if len(full_key) > len(__k):
        del self.nodes[full_key]
    
    # Upgrade n_values, delete everything above the nodes if there's no 
    # other branch
    node = None
    for i in range(len(__k)):
      partial_key = __k[:i+1]
      next_node = self.nodes[partial_key]
      if node is not None:
        node.n_values -= n_remove_nodes
        if next_node.n_values == n_remove_nodes:
          node_key_layer_id = len(node.key)
          del self.nodes[next_node.key]
          del node.value[partial_key[node_key_layer_id:]]
      node = next_node
  
    self.n -= n_remove_nodes
    return

  def __iter__(self):
    return PartialKeyMappingIterator(self)

  def __len__(self) -> int:
    return self.n

  @classmethod
  def from_dict(cls, x) -> PartialKeyMapping:
    return cls(PartialKeyMapping.dfs_nested_dict(x))

  @staticmethod
  def dfs_nested_dict(x, parent_key: Hashable = tuple()) -> Dict[Hashable, Any]:
    if isinstance(parent_key, str):
      parent_key = (parent_key, )

    result = dict()
    for __k, __v in x.items():
      if isinstance(__k, str):
        key = tuple(parent_key) + (__k, )
      else:
        key = tuple(parent_key) + __k
      if issubclass(type(__v), Mapping):
        result.update(PartialKeyMapping.dfs_nested_dict(__v, key))
      else:
        result.update({key: __v})
    
    return result