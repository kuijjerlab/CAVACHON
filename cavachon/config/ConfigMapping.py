from collections.abc import Mapping as MutableMapping
from typing import Any, Mapping, Iterator

class ConfigMapping(MutableMapping):
  """ConfigMapping

  Parent class for configs of ComponentConfig, IOConfig, ModalityConfig,
  ModalityFileConfig, OptimizerConfig, SampleConfig, TrainingConfig and
  etc. Define the basic interface.
  
  """
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for ConfigMapping. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        config in mapping, where the keys are the attributes of the
        children classes.
    
    """
    self.__keys = config.keys()
    for key, value in config.items():
      setattr(self, key, value)

  def __setitem__(self, __name: str, __value: Any):
    """Implementation for __setitem__ in MutableMapping.

    Parameters
    ----------
    __name: str
        name
    
    __value: Any
        value

    """
    setattr(self, __name, __value)

  def __getitem__(self, __name: str) -> Any:
    """Implementation for __getitem__ in MutableMapping.

    Parameters
    ----------
    __name: str
        name

    Returns
    -------
    Any
        value

    """
    return getattr(self, __name)

  def __iter__(self) -> Iterator[Mapping]:
    """Implementation for __iter__ in MutableMapping.

    Returns
    -------
    Iterator[Mapping]
        iterator for the ConfigMapping.

    """
    return iter(self.__keys)
  
  def __len__(self) -> int:
    """Implementation for __len__ in MutableMapping.

    Returns
    -------
    int
        length of MutableMapping.
    
    """
    return len(self.__keys)
  
  def __repr__(self) -> str:
    """Implementation for __repr__ in MutableMapping.

    Returns
    -------
    str
        string representation of MutableMapping.
    
    """
    result = f'{self.__class__.__name__}({{'
    if len(self) == 0:
      return f'{result}}})'
    for i, (key, value) in enumerate(self.items()):
      result += f'{key}: {value}'
      if i != len(self.__keys) - 1:
        result += ', '
      else:
        result += '})'
    return result