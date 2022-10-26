from cavachon.config.config_mapping.ComponentConfig import ComponentConfig
from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from cavachon.config.config_mapping.DatasetConfig import DatasetConfig
from cavachon.config.config_mapping.TrainingConfig import TrainingConfig
from cavachon.utils.GeneralUtils import GeneralUtils
from typing import Any, List, Mapping

class ModelConfig(ConfigMapping):
  """ModelConfig

  Config for model.

  Attributes
  ----------
  name: str
      name of the model.
  
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

  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for ModelConfig. 

    Parameters
    ----------
    name: str
        name of the model.

    components: List[ComponentConfig]
        list of component configs.

    training: Union[Mapping[str, Any], TrainingConfig]
        training config.

    dataset: Union[Mapping[str, Any], DatasetConfig]
        dataset config

    load_weights: bool, optional
        whether or not to load the pretrained weights before training.
        Defaults to False.

    save_weights: bool, optional
        whether or not to save the weights after training. Defaults to
        True.

    """
    # change default values here
    self.name: str = 'cavachon'
    self.components: List[ComponentConfig] = list()
    self.training: TrainingConfig
    self.dataset: DatasetConfig
    self.load_weights: bool = False
    self.save_weights: bool = True

    super().__init__(
        kwargs, 
        [
          'name', 
          'components',
          'training',
          'dataset',
          'load_weights',
          'save_weights'])
    
    # postprocessing
    self.name = GeneralUtils.tensorflow_compatible_str(self.name)
    if not isinstance(self.training, TrainingConfig):
      self.training = TrainingConfig(**self.training)
    if not isinstance(self.dataset, DatasetConfig):
      self.dataset = DatasetConfig(**self.dataset)