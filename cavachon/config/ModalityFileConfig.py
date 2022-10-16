from cavachon.config.ConfigMapping import ConfigMapping
from typing import Any, List, Mapping

class ModalityFileConfig(ConfigMapping):
  """ModalityFileConfig

  Config for modality.

  Attributes
  ----------
  name: str
      name of the modality.
  
  matrix: str
      file path to the matrix file.

  barcodes: str
      file path to the barcodes file.

  features: str
      file path to the features file.
  
  barcodes_colnames: List[str]
      column names of barcodes DataFrame.
  
  features_colnames: List[str]
      column names of features DataFrame.

  """
  def __init__(self, config: Mapping[str, Any]):
    """Constructor for ModalityFileConfig

    Parameters
    ----------
    config: Mapping[str, Any]:
        modality file config in mapping format.
    
    """
    self.name: str
    self.matrix: str
    self.barcodes: str
    self.features: str
    self.barcodes_colnames: List[str] = list()
    self.features_colnames: List[str] = list()
    super().__init__(config)
    