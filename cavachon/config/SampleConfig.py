from cavachon.config.ConfigMapping import ConfigMapping
from cavachon.config.ModalityFileConfig import ModalityFileConfig
from typing import Any, List, Mapping

class SampleConfig(ConfigMapping):
  """SampleConfig

  Config for sample.

  Attributes
  ----------
  name: str
      name of the sample.
  
  description: List[str]
      description of the samples.

  modalities: List[ModalityFileConfig]
      list of modality file configs associated with the sample.

  """

  def __init__(self, config: Mapping[str, Any]):
    """Constructor for SampleConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        sample config in mapping format.
    
    """
    self.name: str
    self.description: str
    self.modalities: List[ModalityFileConfig] = list()
    super().__init__(config)
    for i in range(len(self.modalities)):
      self.modalities[i] = ModalityFileConfig(self.modalities[i])