from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class OptimizerConfig(ConfigMapping):
  """OptimizerConfig

  Config for optimizer.

  Attributes
  ----------
  name: str
      name of the optimizer.
  
  learning_rate: float
      learning rate of the optimizer.

  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for OptimizerConfig. 

    Parameters
    ----------
    name: str, optional
        name of the optimizer. Defaults to 'adam'

    learning_rate: float, optional
        learning rate of the optimizer. Defaults to 1e-4.
    
    """
    # change default values here
    self.name: str = 'adam'
    self.learning_rate: float = 1e-4
    super().__init__(kwargs, ['name', 'learning_rate'])

    # postprocessing
    self.learning_rate = float(self.learning_rate)