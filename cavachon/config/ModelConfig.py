from cavachon.config.ComponentConfig import ComponentConfig
from cavachon.config.ConfigMapping import ConfigMapping
from cavachon.config.DatasetConfig import DatasetConfig
from cavachon.config.ModalityFileConfig import ModalityFileConfig
from cavachon.config.TrainingConfig import TrainingConfig
from typing import Any, List, Mapping

class ModelConfig(ConfigMapping):
  """ModelConfig

  Config for model.

  Attributes
  ----------
  name: str
      name of the sample.
  
  components: List[ComponentConfig]
      list of component configs.

  training: TrainingConfig
      training config.
  
  dataset: DatasetConfig
      dataset config

  """

  def __init__(self, config: Mapping[str, Any]):
    """Constructor for ModelConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        model config in mapping format.

    """
    self.name: str
    self.components: List[ComponentConfig] = list()
    self.training: TrainingConfig
    self.dataset: DatasetConfig
    super().__init__(config)