from collections.abc import MutableMapping
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
    super().__init__()
    self.__keys = set(config.keys())
    for key, value in config.items():
      setattr(self, key, value)

  def __delitem__(self, __name: str) -> None:
    """Implementation for __delitem__ in MutableMapping.

    Parameters
    ----------
    __name: str
        name

    """
    delattr(self, __name)
    self.__keys = set(filter(lambda x: x != __name, self.__keys))

  def __setitem__(self, __name: str, __value: Any) -> None:
    """Implementation for __setitem__ in MutableMapping.

    Parameters
    ----------
    __name: str
        name
    
    __value: Any
        value

    """
    setattr(self, __name, __value)
    if __name not in self.__keys:
      self.__keys.add(__name)

  def __getitem__(self, __name: str) -> Any:
    """Implementation for __getitem__ in MutableMapping.

    Parameters
    ----------
    __name: str
        name

    Raises
    ------
    KeyError
        if __name not in attributes of ConfigMapping

    Returns
    -------
    Any
        value

    """
    if not hasattr(self, __name):
      raise KeyError(__name)

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