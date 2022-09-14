from cavachon.config.ConfigMapping import ConfigMapping
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
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for OptimizerConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        optimizer config in mapping format.
    
    """
    self.name: str
    self.learning_rate: float
    super().__init__(config)