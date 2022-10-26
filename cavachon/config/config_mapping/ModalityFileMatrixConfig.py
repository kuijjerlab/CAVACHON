from cavachon.config.config_mapping.ConfigMapping import ConfigMapping
from typing import Any, List, Mapping

class ModalityFileMatrixConfig(ConfigMapping):
  """ModalityFileMatrixConfig

  Config for modality matrix.

  Attributes
  ----------
  filename: str
      filename of the matrix.
  
  transpose: bool
      if the matrix is transposed (the matrix is transposed if vars 
      as rows, obs as cols).

  """
  def __init__(self, **kwargs: Mapping[str, Any]):
    """Constructor for ModalityFileMatrixConfig

    Parameters
    ----------
    filename: str
        filename of the matrix.
    
    transpose: bool, optional
        if the matrix is transposed (the matrix is transposed if vars 
        as rows, obs as cols). Defaults to False.

    """
    # change default values here
    filename: str
    transpose: bool = False
    super().__init__(kwargs, ['filename', 'transpose'])