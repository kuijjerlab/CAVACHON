from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.model.Model import Model
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.workflow.FilterModifierHandler import FilterModifierHandler
from collections import OrderedDict
from typing import Optional

import tensorflow as tf

class Workflow:
  def __init__(self, filename: str):
    self.config_parser: ConfigParser = ConfigParser(filename)
    self.modality_ordered_map: ModalityOrderedMap = None
    self.data_loader: Optional[DataLoader] = None
    self.filter_handler: Optional[FilterModifierHandler] = None
    self.preprocess_handler: Optional[OrderedDict()] = None
    self.postprocess_handler: Optional[OrderedDict()] = None
  
    return

  def execute(self) -> None:
    self.modality_ordered_map = ModalityOrderedMap.from_config_parser(self.config_parser)
    self.filter_handler = FilterModifierHandler.from_config_parser(
        self.config_parser,
        Constants.CONFIG_FIELD_MODALITY_FILTER,
        Constants.FILTER_STEP_MAPPING)
    self.preprocess_handler = FilterModifierHandler.from_config_parser(
        self.config_parser,
        Constants.CONFIG_FIELD_MODALITY_PREPROCESS,
        Constants.PREPROCESS_STEP_MAPPING)
    self.postprocess_handler = FilterModifierHandler.from_config_parser(
        self.config_parser,
        Constants.CONFIG_FIELD_MODALITY_POSTPROCESS,
        Constants.POSTPROCESS_STEP_MAPPING)
    self.filter_handler.execute(self.modality_ordered_map)
    #self.data_loader = DataLoader.from_modality_ordered_map(self.modality_ordered_map)
    #self.model = Model(self.modality_ordered_map)
    #self.model.fit(self.data_loader.dataset)
    return