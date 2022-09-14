from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, Mapping

class FilterConfig(ConfigMapping):
  """FilterConfig

  Config for Dataset.

  Attributes
  ----------
  step: str
      filter step.

  kwargs: Mapping[str, Any]
      additional parameters for cavachon.filter.AnnDataFilter
  
  """

  def __init__(self, config: Mapping[str, Any]):
    """Constructor for FilterConfig. 

    Parameters
    ----------
    config: Mapping[str, Any]:
        filter config in mapping format.

    """
    self.step: str
    super().__init__(config)
