from cavachon.dataloader.DataLoader import DataLoader
from cavachon.environment.Constants import Constants
from cavachon.model.Model import Model
from cavachon.modifier.Identity import Identity
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.utils.ReflectionHandler import ReflectionHandler
from collections import OrderedDict
from typing import Optional

import tensorflow as tf

class Workflow:
  def __init__(self, filename: str):
    self.config_parser: ConfigParser = ConfigParser(filename)
    self.modality_ordered_map: ModalityOrderedMap = None
    self.data_loader: Optional[DataLoader] = None
    self.model: tf.keras.Model = None
    self.filter_steps = OrderedDict()
    self.preprocess_steps = OrderedDict()
    self.postprocess_steps = OrderedDict()
  
    return

  def execute(self) -> None:
    self.setup_modifier()
    #elf.modality_ordered_map = ModalityOrderedMap.from_config_parser(self.config_parser)
    #self.data_loader = DataLoader.from_modality_ordered_map(self.modality_ordered_map)
    #self.model = Model(self.modality_ordered_map)
    #self.model.fit(self.data_loader.dataset)
    return

  def setup_modifier(self):
    """TODO: put this into a independent class"""
    filter_steps = OrderedDict()
    preprocess_steps = OrderedDict()
    postprocess_step = OrderedDict()
    for modality_name, config in self.config_parser.config_modality.items():
      config_filter_steps = config.get(Constants.CONFIG_FIELD_MODALITY_FILTER)
      config_preprocess_steps = config.get(Constants.CONFIG_FIELD_MODALITY_PREPROCESS)
      config_postprocess_steps = config.get(Constants.CONFIG_FIELD_MODALITY_POSTPROCESS)

      filter_step_list = []
      for config_filter in config_filter_steps:
        filter_step_cls_name = Constants.FILTER_STEP_MAPPING.get(
            config_filter.get('func'))
        filter_step_cls = ReflectionHandler.get_class_by_name(filter_step_cls_name)
        filter_step_list.append(
            filter_step_cls(
                config_filter.get('name', ''),
                config_filter.get('args', {})))

      preprocess_step_list = []
      for config_preprocess in config_preprocess_steps:
        preprocess_step_cls_name = Constants.PREPROCESS_STEP_MAPPING.get(
            config_preprocess.get('func'))
        preprocess_step_cls = ReflectionHandler.get_class_by_name(preprocess_step_cls_name)
        preprocess_step_list.append(
            preprocess_step_cls(
                config_filter.get('name', ''),
                config_filter.get('args', {})))

      if len(preprocess_step_list) == 0:
        preprocess_default_config = {
            'name': 'default', 
            'args': {
                'modality_name': modality_name,
                'target_name': Constants.TENSOR_NAME_X,
                'is_preprocess': True }}
        preprocess_step_list = [Identity(**preprocess_default_config)]
      
      postprocess_step_list = []
      for config_postprocess in config_postprocess_steps:
        postprocess_step_cls_name = Constants.POSTPROCESS_STEP_MAPPING.get(
            config_postprocess.get('func'))
        postprocess_step_cls = ReflectionHandler.get_class_by_name(postprocess_step_cls_name)
        postprocess_step_list.append(
            postprocess_step_cls(
                config_filter.get('name', ''),
                config_filter.get('args', {}))) 
      
      if len(postprocess_step_list) == 0:
        postprocess_default_config = {
            'name': 'default', 
            'args': {
                'modality_name': modality_name,
                'target_name': Constants.TENSOR_NAME_X,
                'is_preprocess': False }}
        postprocess_step_list = [Identity(**postprocess_default_config)]
      
      filter_steps.setdefault(modality_name, filter_step_list)
      preprocess_steps.setdefault(modality_name, preprocess_step_list)
      postprocess_step.setdefault(modality_name, postprocess_step_list)