from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class FilterStep(ABC):
  """TODO: remove setter"""

  def __init__(self, name, args):
    self._name = name
    self._args = args

  @property
  def name(self) -> str:
    return self._name
 
  @name.setter
  def name(self, name) -> None:
    self._name = name
    return

  @property
  def args(self) -> Any:
    return self._args
  
  @args.setter
  def args(self, args) -> None:
    self._args = dict() if args is None else args
    return 

  @abstractmethod
  def execute(self, modality: Modality) -> None:
    pass