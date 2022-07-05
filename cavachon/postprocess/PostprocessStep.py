from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

class PostprocessStep(ABC):

  def __init__(self, name: str, modality_name: str):
    self._name = name
    self._modality_name = modality_name

  @property
  def name(self):
    return self._name
 
  @name.setter
  def name(self, name):
    self._name = name
  
  @property
  def modality_name(self):
    return self._modality_name
 
  @modality_name.setter
  def name(self, modality_name):
    self._modality_name = modality_name
    
  @abstractmethod
  def execute(self, inputs: Dict[str, tf.Tensor], **kwargs) -> Dict[str, tf.Tensor]:
    return