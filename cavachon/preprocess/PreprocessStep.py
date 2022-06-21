from __future__ import annotations
from abc import ABC, abstractmethod

class PreprocessStep(ABC):

  @abstractmethod
  def __init__(self, name, kwargs):
    self._name = name
    self._kwargs = kwargs

  @property
  def name(self):
    return self._name
 
  @name.setter
  def name(self, name):
    self._name = name

  @property
  def kwargs(self):
    return self._kwargs
  
  @kwargs.setter
  def kwargs(self, kwargs):
    self._kwargs = dict() if kwargs is None else kwargs

  @abstractmethod
  def execute(self, modality: Modality) -> None:
    pass