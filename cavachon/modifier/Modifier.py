from abc import ABC, abstractmethod
from typing import Any

class Modifier(ABC): 
  @abstractmethod
  def execute(self, *args, **kwargs) -> Any:
    return