from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from copy import deepcopy
from typing import Any, Mapping

import os

class IOConfig(ConfigMapping):
  """ComponentConfig

  Config for inputs and outputs.

  Attributes
  ----------
  checkpointdir: str
      path to the checkpoint directory.

  datadir: str
      path to the data directory.

  outdir: str
      path to the output directory.

  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for IOConfig. 

    Parameters
    ----------
    checkpointdir: str
        path to the checkpoint directory.

    datadir: str
        path to the data directory.

    outdir: str
        path to the output directory.
    
    """
    # change default values here
    self.checkpointdir: str = './'
    self.datadir: str = './'
    self.outdir: str = './'
    
    # preprocess
    kwargs = deepcopy(kwargs)
    for attributes in kwargs.keys():
      path = kwargs.get(attributes)
      kwargs[attributes] =  os.path.realpath(os.path.dirname(f'{path}/'))

    super().__init__(kwargs, ['checkpointdir', 'datadir', 'outdir'])