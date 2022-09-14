from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class IOConfig(ConfigMapping):
  """ComponentConfig

  Config for inputs and outputs.

  Attributes
  ----------
  datadir: str
      path to the data directory.

  """
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for IOConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        inputs and outputs config in mapping format.
    
    """
    self.datadir: str
    super().__init__(config)