from cavachon.config.ConfigMapping import ConfigMapping
from cavachon.config.FilterConfig import FilterConfig
from typing import Any, List, Mapping

class ModalityConfig(ConfigMapping):
  """ModalityConfig

  Config for modality.

  Attributes
  ----------
  name: str
      name of the modality.
  
  samples: List[str]
      names of the samples.

  type: str
      modality type.

  dist: str
      distribution name of the modality.

  filter: FilterConfig
      filter step configs for the modality.

  """
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for ModalityConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        modality config in mapping format.
    
    """
    self.name: str
    self.samples: List[str] = list()
    self.type: str
    self.dist: str
    self.filter: List[FilterConfig] = list()
    super().__init__(config)
    