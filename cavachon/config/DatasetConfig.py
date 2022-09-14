from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class DatasetConfig(ConfigMapping):
  """DatasetConfig

  Config for Dataset.

  Attributes
  ----------
  batch_size: int
      bath size for iterating dataset.

  """

  def __init__(self, config: Mapping[str, Any]):
    """Constructor for DatasetConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        dataset config in mapping format.

    """
    self.batch_size: int
    super().__init__(config)