from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class DatasetConfig(ConfigMapping):
  """DatasetConfig

  Config for Dataset.

  Attributes
  ----------
  batch_size: int
      bath size for iterating dataset.

  """

  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for DatasetConfig. 

    Parameters
    ----------
    batch_size: int, optional
        bath size for iterating dataset. Defaults to 128

    shuffle: bool, optional
        whether or not to shuffle the dataset during training. Defaults
        to False.

    """
    # change default values here
    self.batch_size: int = 128
    self.shuffle: bool = False
    super().__init__(kwargs, ['batch_size', 'shuffle'])