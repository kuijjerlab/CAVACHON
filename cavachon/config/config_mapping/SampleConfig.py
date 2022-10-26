from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.config.config_mapping.ModalityFileConfig import ModalityFileConfig
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

  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for SampleConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        sample config in mapping format.
    
    """
    # change default values here
    self.name: str
    self.description: str = ''
    self.modalities: List[ModalityFileConfig] = list()
    
    super().__init__(kwargs, ['name', 'description', 'modalities'])
    
    # postprocessing
    for i in range(len(self.modalities)):
      self.modalities[i] = ModalityFileConfig(**self.modalities[i])