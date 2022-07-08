from __future__ import annotations
from cavachon.environment.Constants import Constants
from cavachon.filter.FilterStep import FilterStep
from cavachon.modality.ModalityOrderedMap import ModalityOrderedMap
from cavachon.modifier.Identity import Identity
from cavachon.modifier.TensorModifier import TensorModifier
from cavachon.parser.ConfigParser import ConfigParser
from cavachon.utils.ReflectionHandler import ReflectionHandler
from collections import OrderedDict
from typing import Dict, List, Union

import tensorflow as tf

class FilterModifierHandler:
  def __init__(self, steps: OrderedDict[str, List[Union[FilterStep, TensorModifier]]]):
    self.steps: OrderedDict[str, List[Union[FilterStep, TensorModifier]]] = steps

  @classmethod
  def from_config_parser(cls, config_parser: ConfigParser, field: str, mapping: Dict[str, str]):
    steps = OrderedDict()
    for modality_name, config_modality in config_parser.config_modality.items():
      config_steps = config_modality.get(field)
      step_list = []
      for config in config_steps:
        step_cls_name = mapping.get(config.get('func'))
        step_cls = ReflectionHandler.get_class_by_name(step_cls_name)
        step_list.append(
          step_cls(
              config.get('name', ''),
              config.get('args', {})))

        if issubclass(step_cls, TensorModifier) and len(step_list) == 0:
          is_preprocess = (field == Constants.CONFIG_FIELD_MODALITY_PREPROCESS)
          default_config = {
              'name': 'default', 
              'args': {
                  'modality_name': modality_name,
                  'target_name': Constants.TENSOR_NAME_X,
                  'is_preprocess': is_preprocess }
          }
          step_list.append(
            Identity(**default_config)
          )
      steps.setdefault(modality_name, step_list)
    
    return cls(steps)

  def execute(self, target: Union[ModalityOrderedMap, Dict[str, tf.Tensor]]):
    for modality_name, step_list in self.steps.items():
      for step in step_list:
        if issubclass(step.__class__, FilterStep):
          modality = target.data.get(modality_name)
          step.execute(modality)
        else:
          step.execute(target)
    
    return target