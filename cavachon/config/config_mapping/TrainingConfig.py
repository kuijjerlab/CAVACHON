from cavachon.config.config_mapping.OptimizerConfig import OptimizerConfig
from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class TrainingConfig(ConfigMapping):
  """TrainingConfig

  Config for training.

  Attributes
  ----------
  optimizer: OptimizerConfig
      config for optimizer.
  
  max_n_epochs: int
      maximum number of epochs for training.

  train: bool
      whether or not to retrain (finetune) the model.

  early_stopping: bool
      whether or not to use early stopping when training the model. 
      Ignored if `train=False`.
  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for TrainingConfig. 

    Parameters
    ----------
    optimizer: MutableMapping[str, Any]
        config for optimizer in MutableMapping format.

    max_n_epochs: int, optional
        maximum number of epochs for training. Defaults to 500.

    train: bool, optional
        whether or not to retrain (finetune) the model. Defaults to 
        True.

    early_stopping: bool, optional
        whether or not to use early stopping when training the model. 
        Ignored if `train=False`. Defaults to True.
    """
    # change default values here
    self.optimizer: OptimizerConfig
    self.max_n_epochs: int = 500
    self.train: bool = True
    self.early_stopping: bool = True
    super().__init__(kwargs, ['optimizer', 'max_n_epochs', 'train', 'early_stopping'])
    
    # postprocessing
    self.setdefault(
        'optimizer', 
        {'name': 'adam', 'learning_rate': 1e-4})
    self.optimizer = OptimizerConfig(**self.optimizer)    