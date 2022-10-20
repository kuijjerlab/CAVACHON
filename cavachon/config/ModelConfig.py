from cavachon.config.ComponentConfig import ComponentConfig
from cavachon.config.ConfigMapping import ConfigMapping
from cavachon.config.DatasetConfig import DatasetConfig
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

  load_weights: bool
      whether or not to load the pretrained weights before training.

  save_weights: bool
      whether or not to save the weights after training.
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
    self.load_weights: bool
    self.save_weights: bool
    super().__init__(config)