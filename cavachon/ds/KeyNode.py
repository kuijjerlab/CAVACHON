from typing import Any, Dict, Hashable

class KeyNode(Hashable):
  def __init__(self, key: Hashable):
    self.key: Hashable = key
    self.n_values = 0
    self.value = None
  
  def __hash__(self) -> int:
    return hash(self.key)

  def __repr__(self) -> str:
    return f"{{{self.key}: {self.value}}}"
  
class PartialKeyNode(KeyNode):
  def __init__(self, key: Hashable):
    super().__init__(key)
    self.value: Dict[Hashable, KeyNode] = dict()

  @property
  def values_in_subtree(self) -> Dict[Hashable, Any]:
    result = dict()
    for node in self.value.values():
      result.update(node.values_in_subtree)
    return result

class FullKeyNode(KeyNode):
  def __init__(self, key: Hashable, value: any):
    super().__init__(key)
    self.value: Any = value

  @property
  def values_in_subtree(self) -> Dict[Hashable, Any]:
    return { self.key: self.value }
# %%
