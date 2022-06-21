
from cavachon.dataloader.DataLoader import DataLoader
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.parser.ConfigParser import ConfigParser
from typing import Optional

class Workflow:
  def __init__(self, filename: str):
    self.config_parser: ConfigParser = ConfigParser(filename)
    self.modality_ordered_map: ModalityOrderedMap = None
    self.data_loader: Optional[DataLoader] = None
  
    return

  def execute(self) -> None:
    self.modality_ordered_map = ModalityOrderedMap.from_config_parser(self.config_parser)
    self.modality_ordered_map.preprocess()
    self.data_loader = DataLoader.from_modality_ordered_map(self.modality_ordered_map)
    return
