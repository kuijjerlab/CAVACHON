from abc import ABC, abstractmethod
from cavachon.environment.Constants import Constants
from typing import Dict

import tensorflow as tf

class TensorModifier(ABC):
  """TODO: remove setter, maybe could refactor to something similar to FilterStep.py"""
  """TODO: modality_name is not needed"""
  def __init__(self, name, args):
    self._name = name
    self._modality_name = args.get('modality_name')
    self._target_name = args.get('target_name')
    self._is_preprocess = args.get('is_preprocess')
    if self._is_preprocess:
      self._postfix = Constants.CONFIG_FIELD_MODALITY_PREPROCESS
    else:
      self._postfix = Constants.CONFIG_FIELD_MODALITY_POSTPROCESS

  @property
  def name(self) -> str:
    return self._name
 
  @name.setter
  def name(self, name) -> None:
    self._name = name
    return
  
  @property
  def modality_name(self) -> str:
    return self._modality_name
 
  @modality_name.setter
  def name(self, modality_name) -> None:
    self._modality_name = modality_name
    return

  @property
  def target_name(self) -> str:
    return self._target_name

  @target_name.setter
  def target_name(self, target_name) -> None:
    self._target_name = target_name
    return
    
  @property
  def is_preprocess(self) -> bool:
    return self.is_preprocess
  
  @is_preprocess.setter
  def is_preprocess(self, is_preprocess: bool) -> None:
    self._is_preprocess = is_preprocess
    return

  @property
  def postfix(self) -> str:
    return self._postfix

  @postfix.setter
  def postfix(self, postfix) -> None:
    self._postfix = postfix
    return
  
  @abstractmethod
  def execute(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return